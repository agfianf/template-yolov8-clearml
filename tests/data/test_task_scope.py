"""Resolving "which projects / which tasks" into two lists of task ids.

The case worth protecting is the leak: a project is named for training and one
of *its own* tasks is named for testing. Listing task ids by hand, that task
ends up in both, and every metric on the test split is then reported against
images the model was trained on.
"""

import logging

import pytest

from src.data.task_scope import resolve_task_scope


PROJECTS = {
    # project id -> tasks, as CVAT returns them
    53: [
        {"id": 1186, "name": "B8", "size": 69, "status": "annotation"},
        {"id": 1181, "name": "B7", "size": 147, "status": "annotation"},
        {"id": 1169, "name": "B6", "size": 73, "status": "completed"},
    ],
    46: [
        {"id": 1070, "name": "B2", "size": 209, "status": "completed"},
        {"id": 1069, "name": "B1", "size": 0, "status": "annotation"},
    ],
    99: [],
}


def list_tasks(project_id: int) -> list[dict]:
    return sorted(PROJECTS[project_id], key=lambda t: t["id"])


# --------------------------------------------------------------------------
# Expanding projects
# --------------------------------------------------------------------------


def test_a_project_expands_to_all_of_its_tasks() -> None:
    scope = resolve_task_scope(
        list_tasks, task_ids_train=[], task_ids_test=[], project_ids_train=[53]
    )

    assert scope.train == [1169, 1181, 1186]
    assert scope.test == []


def test_empty_tasks_are_skipped() -> None:
    """A task with no frames exports an empty archive and fails later, obscurely."""
    scope = resolve_task_scope(
        list_tasks, task_ids_train=[], task_ids_test=[], project_ids_train=[46]
    )

    assert scope.train == [1070]


def test_a_task_without_a_size_field_is_kept() -> None:
    """Unknown size is not zero size.

    Guards the v2 path and any future CVAT version: if the field stops being
    returned, dropping every task would look exactly like an empty project.
    """
    tasks = [{"id": 5, "name": "no-size", "status": "completed"}]

    scope = resolve_task_scope(
        lambda _pid: tasks, task_ids_train=[], task_ids_test=[], project_ids_train=[1]
    )

    assert scope.train == [5]


def test_projects_and_explicit_tasks_add_up() -> None:
    scope = resolve_task_scope(
        list_tasks, task_ids_train=[900], task_ids_test=[], project_ids_train=[46]
    )

    assert scope.train == [900, 1070]


def test_a_task_listed_twice_is_downloaded_once() -> None:
    scope = resolve_task_scope(
        list_tasks, task_ids_train=[1070, 1070], task_ids_test=[], project_ids_train=[46]
    )

    assert scope.train == [1070]


# --------------------------------------------------------------------------
# The exclusion -- the reason this module exists
# --------------------------------------------------------------------------


def test_a_test_task_inside_a_training_project_is_not_trained_on() -> None:
    scope = resolve_task_scope(
        list_tasks,
        task_ids_train=[],
        task_ids_test=[1181],  # belongs to project 53
        project_ids_train=[53],
    )

    assert scope.train == [1169, 1186]
    assert scope.test == [1181]
    assert scope.excluded_from_train == [1181]


def test_exclusion_also_applies_to_an_explicitly_listed_train_task() -> None:
    scope = resolve_task_scope(
        list_tasks, task_ids_train=[1070, 1069], task_ids_test=[1069]
    )

    assert scope.train == [1070]
    assert scope.excluded_from_train == [1069]


def test_a_whole_test_project_is_excluded_from_training() -> None:
    scope = resolve_task_scope(
        list_tasks,
        task_ids_train=[],
        task_ids_test=[],
        project_ids_train=[53, 46],
        project_ids_test=[46],
    )

    assert scope.train == [1169, 1181, 1186]
    assert scope.test == [1070]


def test_everything_being_test_is_an_error_not_an_empty_run() -> None:
    with pytest.raises(ValueError, match="no training tasks"):
        resolve_task_scope(
            list_tasks, task_ids_train=[1070], task_ids_test=[1070], project_ids_train=[]
        )


def test_nothing_configured_is_an_error() -> None:
    with pytest.raises(ValueError, match="no training tasks"):
        resolve_task_scope(list_tasks, task_ids_train=[], task_ids_test=[])


def test_a_project_with_no_tasks_warns_and_does_not_crash() -> None:
    with pytest.raises(ValueError, match="no training tasks"):
        resolve_task_scope(
            list_tasks, task_ids_train=[], task_ids_test=[], project_ids_train=[99]
        )


# --------------------------------------------------------------------------
# Log volume
# --------------------------------------------------------------------------


def test_scope_is_summarised_in_one_line(capture_src_logs) -> None:
    with capture_src_logs(logging.INFO) as records:
        resolve_task_scope(
            list_tasks,
            task_ids_train=[],
            task_ids_test=[1181],
            project_ids_train=[53, 46],
        )

    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert "train 3 task(s)" in message
    assert "excluded from train as test: [1181]" in message
    # The unfinished-task count is in the same line: a task still in
    # `annotation` status may be half-labelled, which trains false negatives.
    assert "not yet 'completed'" in message
