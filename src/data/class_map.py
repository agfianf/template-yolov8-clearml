"""One class-id order for every source in a run.

The problem this exists for: a YOLO label file stores a class *index*, but COCO
stores a category *id*, and CVAT hands out category ids per project. Two CVAT
projects that were annotated with the same label names still export them in a
different order, and a project that never saw `speed_limit` simply omits it --
so `category_id - 1` means "car" in one task and "speed_limit" in the next. The
converter merges every task into one directory, so those indices land in the
same dataset and the model is trained on a scrambled target.

The fix is a single name -> index map built once, before any conversion, from
the union of every source's categories. Conversion then looks up by *name*.

Order is either pinned explicitly (`class_names` in `args_data`) or derived by
sorting the union. Sorting is what makes it reproducible: a second run over the
same tasks, in any order, produces the same map. It is *not* stable against
adding a class -- a new name sorts into the middle and shifts everything after
it. Pin `class_names` before fine-tuning on top of an existing checkpoint.
"""

import json

from dataclasses import dataclass, field

from src.utils.logging import Tally, get_logger


logger = get_logger(__name__)


class UnknownClassError(Exception):
    """A source contains a class that the pinned `class_names` does not list."""


def _key(name: str) -> str:
    """Normalize a label for matching.

    Case and surrounding whitespace differ between projects more often than
    anyone intends -- "Speed_Limit" and "speed_limit " are the same label to
    everyone except a dict lookup.
    """
    return name.strip().lower()


def read_category_names(json_path: str) -> list[str]:
    """Return the category names of a COCO file, in file order.

    Reads the whole JSON, which is what the converter does moments later
    anyway. Splitting out a streaming parser to save one pass would buy a few
    hundred milliseconds per task and cost a dependency.
    """
    with open(json_path) as f:
        payload = json.load(f)
    return [cat["name"] for cat in payload.get("categories", [])]


@dataclass
class ClassMap:
    """Name -> class index, shared by every source in a run.

    `names` is the index order itself: `names[i]` is the label written as `i`
    in the YOLO files and as entry `i` of `data.yaml`.
    """

    names: list[str]
    _by_key: dict[str, int] = field(init=False, repr=False)
    unknown: Tally = field(default_factory=Tally, repr=False)

    def __post_init__(self) -> None:
        self._by_key = {_key(name): i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int | None:
        """Class index for `name`, or None if it is not in the map."""
        return self._by_key.get(_key(name))

    def describe(self) -> str:
        return ", ".join(f"{i}:{name}" for i, name in enumerate(self.names))


def build_class_map(
    annotation_paths: list[str],
    explicit_names: list[str] | None = None,
    exclude_class: list[str] | None = None,
) -> ClassMap:
    """Build the run's canonical class order.

    Parameters
    ----------
    annotation_paths
        Every `instances_default.json` in the run -- train *and* test. The test
        split has to be indexed by the same map or its labels mean something
        else than the training labels.
    explicit_names
        Pinned order. Takes precedence over the sources; a name listed here but
        present in no source keeps its slot, which is exactly what makes a
        pinned list survive a task that is missing a class.
    exclude_class
        Names dropped from the map entirely. Their annotations are already
        filtered out upstream, so leaving them in would only reserve an index
        for a class with no instances -- and inflate `nc`.

    """
    excluded = {_key(name) for name in (exclude_class or []) if name.strip()}

    if explicit_names:
        names = [n.strip() for n in explicit_names if n.strip()]
        source = "pinned"
    else:
        # First spelling seen wins for display; matching is case-insensitive.
        seen: dict[str, str] = {}
        for path in annotation_paths:
            for name in read_category_names(path):
                seen.setdefault(_key(name), name.strip())
        names = [seen[k] for k in sorted(seen)]
        source = "derived"

    names = [n for n in names if _key(n) not in excluded]

    # Dedupe, keeping first occurrence: a pinned list is hand-edited, and a
    # repeated name would silently make one of the two entries unreachable.
    deduped: list[str] = []
    for name in names:
        if _key(name) not in {_key(n) for n in deduped}:
            deduped.append(name)
        else:
            logger.warning("class_names lists %r more than once, keeping the first", name)

    if not deduped:
        raise ValueError(
            "class map is empty: no categories found in "
            f"{len(annotation_paths)} annotation file(s) after excluding "
            f"{sorted(excluded) or 'nothing'}"
        )

    class_map = ClassMap(names=deduped)
    logger.info(
        "class map (%s): %d classes -> %s", source, len(class_map), class_map.describe()
    )

    if explicit_names:
        _warn_about_unpinned(annotation_paths, class_map, excluded)

    return class_map


def warn_if_orders_disagree(annotation_paths: list[str]) -> bool:
    """Warn when sources disagree on category order, and say so in one line.

    Only reachable with `unify_class_order` off, where the disagreement is
    silently baked into the labels. Returns whether anything disagreed, for
    tests.
    """
    orders = {}
    for path in annotation_paths:
        orders.setdefault(tuple(_key(n) for n in read_category_names(path)), path)
    if len(orders) <= 1:
        return False
    logger.warning(
        "unify_class_order is off and %d source(s) disagree on category order, so "
        "the same class index means different things in different tasks: %s",
        len(orders),
        " | ".join(
            f"{path}: {', '.join(order)}" for order, path in list(orders.items())[:3]
        ),
    )
    return True


def _warn_about_unpinned(
    annotation_paths: list[str], class_map: ClassMap, excluded: set[str]
) -> None:
    """Name anything a source has that the pinned list does not.

    Warned about once here, at build time, rather than per annotation during
    conversion -- by then the count is in the thousands and the useful fact
    (which label, which task) is the same every time.
    """
    missing = Tally()
    for path in annotation_paths:
        for name in read_category_names(path):
            if _key(name) in excluded:
                continue
            if class_map.index_of(name) is None:
                missing.add(name, path)
    if missing:
        logger.warning(
            "not in the pinned class_names, and not excluded: %s", missing.summary()
        )
