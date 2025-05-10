from rich import print
from ultralytics import settings


def init_ultralytics_settings():
    """Initialize YOLO settings for ClearML integration."""
    # Disable ClearML integration in Ultralytics settings
    print("ClearML integration disabled in Ultralytics settings.")
    settings.update({"clearml": False})
    print("Settings Ultralytics ClearML:", settings["clearml"])
