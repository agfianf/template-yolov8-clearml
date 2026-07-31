"""Turn "which projects / which tasks" into the two lists of task ids to fetch.

CVAT downloads happen per *task*, but people think in *projects*: "train on
everything in project 53, except the batch I set aside for testing". Listing
every task id by hand is where that goes wrong -- a new task added in CVAT is
silently left out of training, and a task used for testing is easy to leave in
the training list too, which leaks the test set into training and inflates
every metric on it.

This module expands project ids into task ids and subtracts the test tasks from
the training set, so the exclusion happens once, in one place, rather than in
whoever remembers to edit both lists.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.utils.logging import get_logger


logger = get_logger(__name__)

# CVAT task states. Anything that is not `completed` may be partially
# annotated, and a partially annotated image teaches the model that a real
# object is background -- worth naming in the log, not worth silently dropping.
COMPLETED = "completed"


@dataclass
class TaskScope:
    """The task ids to download, and how they were arrived at."""

    train: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    excluded_from_train: list[int] = field(default_factory=list)


def _dedupe(ids: list[int]) -> list[int]:
    """Preserve order, drop repeats -- a task listed twice downloads twice."""
    seen = set()
    out = []
    for task_id in ids:
        if task_id not in seen:
            seen.add(task_id)
            out.append(task_id)
    return out


def _expand(
    project_ids: list[int],
    list_tasks: Callable[[int], list[dict]],
    role: str,
) -> tuple[list[int], list[str]]:
    """Expand project ids into task ids, and describe what was found."""
    task_ids: list[int] = []
    notes: list[str] = []
    for project_id in project_ids:
        tasks = list_tasks(project_id)
        # A task with no frames exports an empty archive and then fails in the
        # converter with a much less obvious message.
        usable = [t for t in tasks if (t.get("size") or 0) > 0]
        empty = len(tasks) - len(usable)
        unfinished = sum(1 for t in usable if t.get("status") != COMPLETED)

        if not tasks:
            logger.warning("cvat project %s (%s) has no tasks", project_id, role)
        elif not usable:
            logger.warning(
                "cvat project %s (%s): all %d task(s) are empty",
                project_id,
                role,
                len(tasks),
            )

        task_ids.extend(t["id"] for t in usable)
        note = f"project {project_id}: {len(usable)} task(s)"
        if empty:
            note += f", {empty} empty skipped"
        if unfinished:
            note += f", {unfinished} not yet 'completed'"
        notes.append(note)
    return task_ids, notes


def resolve_task_scope(
    list_tasks: Callable[[int], list[dict]],
    task_ids_train: list[int],
    task_ids_test: list[int],
    project_ids_train: list[int] | None = None,
    project_ids_test: list[int] | None = None,
) -> TaskScope:
    """Resolve the configured projects and tasks into two task id lists.

    Parameters
    ----------
    list_tasks
        `project_id -> [task dict]`, i.e. a downloader's
        `list_tasks_of_project`. Injected rather than imported so this is
        testable without CVAT.
    task_ids_train, task_ids_test
        Explicitly listed tasks. Kept, and combined with whatever the projects
        expand to -- the two ways of naming sources are additive, so a project
        plus one extra task from elsewhere is expressible.
    project_ids_train, project_ids_test
        Projects to take every task from.

    Returns
    -------
    TaskScope
        `train` with the test tasks removed, `test`, and which ids that removal
        took out -- which is the number worth seeing, because a test task that
        was also in the training project is exactly the leak this prevents.

    """
    train_from_projects, train_notes = _expand(
        project_ids_train or [], list_tasks, "train"
    )
    test_from_projects, test_notes = _expand(project_ids_test or [], list_tasks, "test")

    test = _dedupe(list(task_ids_test or []) + test_from_projects)
    train_all = _dedupe(list(task_ids_train or []) + train_from_projects)

    # The rule: a task used for testing is never trained on, however it got
    # into the training list. Expanding a project is what makes this necessary
    # -- the test task usually lives *inside* the training project.
    test_set = set(test)
    train = [task_id for task_id in train_all if task_id not in test_set]
    excluded = [task_id for task_id in train_all if task_id in test_set]

    detail = "; ".join(train_notes + test_notes)
    logger.info(
        "cvat scope: train %d task(s), test %d task(s)%s%s",
        len(train),
        len(test),
        f" ({detail})" if detail else "",
        f"; {len(excluded)} excluded from train as test: {excluded}" if excluded else "",
    )

    if not train:
        raise ValueError(
            "no training tasks: task_ids_train and project_ids_train resolved to "
            f"{train_all or 'nothing'}"
            + (f", and all of it is in task_ids_test ({test})" if excluded else "")
        )

    return TaskScope(train=train, test=test, excluded_from_train=excluded)
