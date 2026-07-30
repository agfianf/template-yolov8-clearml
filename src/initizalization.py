from ultralytics import settings

from src.utils.logging import get_logger


logger = get_logger(__name__)


def init_ultralytics_settings():
    """Initialize YOLO settings for ClearML integration."""
    # Disable ClearML integration in Ultralytics settings; we log it ourselves
    # from src/yolov8/callbacks.py.
    settings.update({"clearml": False})
    logger.debug("ultralytics clearml integration: %s", settings["clearml"])
