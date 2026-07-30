from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel

from src.utils.logging import Tally, get_logger


logger = get_logger(__name__)


@dataclass
class FilterReport:
    """What annotation filtering did, as counts rather than as a line per item."""

    total: int = 0
    kept: int = 0
    dropped_area: int = 0
    dropped_class: Tally = field(default_factory=Tally)
    dropped_attr: Tally = field(default_factory=Tally)

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    def reasons(self, area_segment_min: float | None = None) -> str:
        parts = []
        if self.dropped_class:
            parts.append(
                f"class {self.dropped_class.total():,} [{self.dropped_class.summary()}]"
            )
        if self.dropped_area:
            parts.append(f"area<{area_segment_min} = {self.dropped_area:,}")
        if self.dropped_attr:
            parts.append(f"attr {self.dropped_attr.summary()}")
        return ", ".join(parts)


class License(BaseModel):
    id: int
    name: str
    url: str


class Info(BaseModel):
    year: str
    version: str
    description: str
    contributor: str
    url: str
    date_created: str


class Category(BaseModel):
    id: int
    name: str
    supercategory: str | None


class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flickr_url: str | None
    coco_url: str | None
    date_captured: int


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]]
    area: float
    bbox: list[float]
    iscrowd: int
    attributes: dict


class Coco(BaseModel):
    licenses: list[License]
    info: Info
    categories: list[Category]
    images: list[Image]
    annotations: list[Annotation] | None

    def get_categoryid_to_namecat(
        self,
        exclude_class: list[str] = None,  # noqa: ARG002
    ) -> dict[int, str]:
        categories_map = {}
        # exclude_class  = [lbl.lower() for lbl in exclude_class]  # noqa: ERA001
        # if len(exclude_class) > 0:
        #     print("WE EXCLUDE CLASS:", exclude_class)  # noqa: ERA001

        for cat in self.categories:
            # if cat.name not in exclude_class:
            categories_map[cat.id] = cat.name
        return categories_map

    def get_imageid_to_image(self) -> dict[int, Image]:
        image_map = {}
        for img in self.images:
            image_map[img.id] = img
        return image_map

    def get_imageid_to_annotations(  # noqa: C901
        self,
        exclude_class: list[str] = None,
        attributes_excluded: dict[str, str] = None,
        area_segment_min: float = None,
    ) -> tuple[dict[int, list[Annotation]], FilterReport]:
        """Return image_id -> annotations, plus a report of what was filtered out.

        Filters are applied at the annotation level. The per-annotation decisions
        are tallied and logged as one INFO line; individual decisions go to DEBUG,
        because there is one per annotation and a dataset has tens of thousands.
        """
        if exclude_class is None:
            exclude_class = []

        imageid2anns = defaultdict(list)
        exclude_class = [lbl.lower() for lbl in exclude_class]
        id2label = self.get_categoryid_to_namecat()
        logger.debug("categories: %s", id2label)

        report = FilterReport(total=len(self.annotations or []))

        for ann in self.annotations or []:
            skip = False
            if area_segment_min is not None and ann.area < area_segment_min:
                logger.debug(
                    "ann %s dropped: area %s < %s", ann.id, ann.area, area_segment_min
                )
                report.dropped_area += 1
                continue

            if attributes_excluded is not None:
                for attr_name, attr_value in attributes_excluded.items():
                    attributes_dataset = set(
                        ann.attributes.get(attr_name)
                        .replace(", ", ",")
                        .replace(" ,", ",")
                        .split(",")
                    )
                    attributes_config = set(
                        attr_value.replace(", ", ",").replace(" ,", ",").split(",")
                    )
                    if attributes_dataset is None:
                        continue

                    if isinstance(attributes_dataset, set):
                        intersection = attributes_config.intersection(attributes_dataset)
                        if intersection:
                            logger.debug(
                                "ann %s dropped: attribute %s in %s",
                                ann.id,
                                attr_name,
                                sorted(intersection),
                            )
                            report.dropped_attr.add(attr_name, ann.id)
                            skip = True
                            break
                        break

            if id2label[ann.category_id] in exclude_class:
                logger.debug(
                    "ann %s dropped: class %s excluded", ann.id, id2label[ann.category_id]
                )
                report.dropped_class.add(id2label[ann.category_id], ann.id)
                skip = True

            if skip:
                continue

            report.kept += 1
            imageid2anns[ann.image_id].append(ann)

        reasons = report.reasons(area_segment_min)
        logger.info(
            "annotations: %s total -> %s kept, %s dropped%s",
            f"{report.total:,}",
            f"{report.kept:,}",
            f"{report.dropped:,}",
            f" ({reasons})" if reasons else "",
        )
        return imageid2anns, report

    def checking_task(self):
        task_type = set()
        if len(self.annotations) == 0:
            logger.warning("no annotations in this COCO file")
            return list(task_type)

        for ann in self.annotations:
            if len(ann.bbox) != 0:
                task_type.add("detection")
            if len(ann.segmentation) != 0:
                task_type.add("segmentation")
        return list(task_type)
