from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# fmt: off
class AugmentParams(BaseModel):
    hsv_h: float = Field(0.015, description="image HSV-Hue augmentation (fraction)")
    hsv_s: float = Field(0.7, description="image HSV-Saturation augmentation (fraction)")
    hsv_v: float = Field(0.4, description="image HSV-Value augmentation (fraction)")
    degrees: float = Field(25.0, description="image rotation (+/- deg)")
    translate: float = Field(0.1, description="image translation (+/- fraction)")
    scale: float = Field(0.5, description="image scale (+/- gain)")
    shear: float = Field(0.0, description="image shear (+/- deg)")
    perspective: float = Field(0.0, description="image perspective (+/- fraction), range 0-0.001")  # noqa: E501
    flipud: float = Field(0.2, description="image flip up-down (probability)")
    fliplr: float = Field(0.5, description="image flip left-right (probability)")
    mosaic: float = Field(1.0, description="image mosaic (probability)")
    mixup: float = Field(0.0, description="image mixup (probability)")
    copy_paste: float = Field(0.0, description="segment copy-paste (probability)")


class ExportFormatParams(BaseModel):
    torchscript: bool = Field(True, description="TorchScript")
    onnx: bool = Field(True, description="ONNX")
    openvino: bool = Field(0, description="OpenVINO")
    engine: bool = Field(0, description="TensorRT")
    coreml: bool = Field(0, description="CoreML")
    saved_model: bool = Field(0, description="TensorFlow SavedModel")
    pb: bool = Field(0, description="TensorFlow GraphDef")
    tflite: bool = Field(0, description="TensorFlow Lite")
    edgetpu: bool = Field(0, description="TensorFlow Edge TPU")
    tfjs: bool = Field(0, description="TensorFlow.js")
    paddle: bool = Field(0, description="PaddlePaddle")


class ExportParams(BaseModel):
    keras: bool = Field(False, description="Export as Keras model")
    optimize: bool = Field(False, description="Optimize model")
    half: bool = Field(True, description="Use half precision")
    int8: bool = Field(False, description="Use int8 quantization")
    dynamic: bool = Field(False, description="Enable dynamic axes")
    simplify: bool = Field(False, description="Simplify model")
    opset: int | None = Field(None, description="ONNX opset version")
    workspace: int = Field(4, description="TensorRT workspace size")
    nms: bool = Field(False, description="Export with NMS")


class ExportConfig(BaseModel):
    format: ExportFormatParams = Field(default_factory=ExportFormatParams, description="Export format options")  # noqa: E501
    params: ExportParams = Field(default_factory=ExportParams, description="Export parameters")  # noqa: E501


class LoggingParams(BaseModel):
    project: str = Field("Debug/yolov8", description="Project name for logging")
    name: str = Field("training-yolo", description="Experiment name for logging")


class TaskParams(BaseModel):
    model_name: str = Field("yolov8n", description="Model name")
    model_latest_id: str = Field("", description="Latest model ID")
    # pretrained: Optional[str] = Field(None, description="Pretrained model ID")  # noqa: E501, ERA001


class CVATParams(BaseModel):
    task_ids_train: list[int] = Field(default_factory=lambda: [48], description="CVAT task IDs for training")  # noqa: E501
    task_ids_test: list[int] = Field(default_factory=list, description="CVAT task IDs for testing")  # noqa: E501


# fmt: off
class LabelStudioParams(BaseModel):
    project_id_train: int | None = Field(None, description="Label Studio project ID for training")  # noqa: E501
    project_id_test: int | None = Field(None, description="Label Studio project ID for testing")  # noqa: E501
# fmt: on


class S3Params(BaseModel):
    s3_uri_dir_train: str | None = Field(None, description="S3 URI for training data")
    s3_uri_dir_test: str | None = Field(None, description="S3 URI for testing data")


class DataParamsInner(BaseModel):
    train_ratio: float = Field(0.8, description="Training data ratio")
    val_ratio: float = Field(0.2, description="Validation data ratio")
    test_ratio: float | None = Field(None, description="Test data ratio")


class DataParams(BaseModel):
    cvat: CVATParams = Field(
        default_factory=CVATParams, description="CVAT data parameters"
    )
    label_studio: LabelStudioParams = Field(
        default_factory=LabelStudioParams, description="Label Studio data parameters"
    )
    s3: S3Params = Field(default_factory=S3Params, description="S3 data parameters")
    params: DataParamsInner = Field(
        default_factory=DataParamsInner, description="Data split ratios"
    )
    class_exclude: Any | None = Field(None, description="Classes to exclude")
    attributes_exclude: Any | None = Field(None, description="Attributes to exclude")
    area_segment_min: int = Field(0, description="Minimum area for segments")


class TrainParams(BaseModel):
    augment: bool = Field(True, description="Enable augmentation")
    epochs: int = Field(1000, ge=1, le=10000, description="Number of epochs to train for")
    patience: int = Field(0, ge=0, description="Early stopping patience")
    batch: int = Field(2, ge=1, description="Batch size")
    imgsz: int = Field(640, ge=32, description="Input image size")
    save: bool = Field(True, description="Save checkpoints and results")
    save_period: int = Field(-1, description="Checkpoint save period")
    cache: bool = Field(True, description="Cache data loading")
    device: Any | None = Field(None, description="Device to run on")
    workers: int = Field(8, ge=0, description="Number of worker threads")
    project: str | None = Field(None, description="Project name")
    name: str | None = Field(None, description="Experiment name")
    exist_ok: bool = Field(True, description="Overwrite existing experiment")
    pretrained: bool = Field(True, description="Use pretrained model")
    optimizer: str = Field("auto", description="Optimizer type")
    verbose: bool = Field(True, description="Verbose output")
    seed: int = Field(0, ge=0, description="Random seed")
    deterministic: bool = Field(True, description="Deterministic mode")
    single_cls: bool = Field(False, description="Single-class training")
    rect: bool = Field(False, description="Rectangular training")
    cos_lr: bool = Field(False, description="Cosine learning rate scheduler")
    close_mosaic: int = Field(
        0, description="Disable mosaic augmentation for final epochs"
    )
    resume: bool = Field(False, description="Resume from last checkpoint")
    amp: bool = Field(True, description="Automatic Mixed Precision (AMP) training")
    fraction: float = Field(0.9, gt=0, le=1, description="Dataset fraction to train on")
    profile: bool = Field(False, description="Profile ONNX/TensorRT speeds")
    lr0: float = Field(0.001, gt=0, description="Initial learning rate")
    lrf: float = Field(0.0001, gt=0, description="Final learning rate factor")
    momentum: float = Field(0.937, ge=0, le=1, description="SGD momentum/Adam beta1")
    weight_decay: float = Field(0.0005, ge=0, description="Optimizer weight decay")
    warmup_epochs: float = Field(3.0, ge=0, description="Warmup epochs")
    warmup_momentum: float = Field(0.8, ge=0, le=1, description="Warmup initial momentum")
    warmup_bias_lr: float = Field(0.1, ge=0, description="Warmup initial bias lr")
    box: float = Field(7.5, gt=0, description="Box loss gain")
    cls: float = Field(0.5, gt=0, description="Class loss gain")
    dfl: float = Field(1.5, gt=0, description="DFL loss gain")
    pose: float = Field(12.0, gt=0, description="Pose loss gain")
    kobj: float = Field(2.0, gt=0, description="Keypoint obj loss gain")
    label_smoothing: float = Field(0.0, ge=0, le=1, description="Label smoothing")
    nbs: int = Field(64, ge=1, description="Nominal batch size")
    overlap_mask: bool = Field(True, description="Masks should overlap during training")
    mask_ratio: int = Field(4, ge=1, description="Mask downsample ratio")
    dropout: float = Field(0.0, ge=0, le=1, description="Dropout regularization")
    val: bool = Field(True, description="Validate/test during training")

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, v: int) -> int:
        """Validate that image size is a multiple of 32."""
        if v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v


class ValParams(BaseModel):
    batch: int = Field(1, description="Batch size for validation")
    save_json: bool = Field(False, description="Save results to JSON")
    save_hybrid: bool = Field(False, description="Save hybrid labels")
    conf: float = Field(0.35, description="Object confidence threshold")
    iou: float = Field(0.6, description="IoU threshold for NMS")
    max_det: int = Field(100, description="Max detections per image")
    half: bool = Field(True, description="Use half precision")
    device: Any = Field(0, description="Device for validation")
    dnn: bool = Field(False, description="Use OpenCV DNN for ONNX inference")
    plots: bool = Field(True, description="Show plots during training")
    rect: bool = Field(False, description="Rectangular validation")
    split: Literal["val", "test", "train"] = Field(
        "val", description="Dataset split for validation"
    )  # noqa: E501


class Params(BaseModel):
    augment: AugmentParams = Field(
        default_factory=AugmentParams, description="Augmentation parameters"
    )
    export: ExportConfig = Field(
        default_factory=ExportConfig, description="Export configuration"
    )
    logging: LoggingParams = Field(
        default_factory=LoggingParams, description="Logging parameters"
    )
    task: TaskParams = Field(default_factory=TaskParams, description="Task parameters")
    data: DataParams = Field(default_factory=DataParams, description="Data parameters")
    train: TrainParams = Field(
        default_factory=TrainParams, description="Training parameters"
    )
    val: ValParams = Field(default_factory=ValParams, description="Validation parameters")


# fmt: on

params = Params()
