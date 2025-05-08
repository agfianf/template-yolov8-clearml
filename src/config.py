# https://docs.ultralytics.com/usage/cfg/#train
# https://docs.ultralytics.com/usage/cfg/#augmentation
args_augment = {
    "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
    "degrees": 25.0,  # image rotation (+/- deg)
    "translate": 0.1,  # image translation (+/- fraction)
    "scale": 0.5,  # image scale (+/- gain)
    "shear": 0.0,  # image shear (+/- deg)
    "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.2,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 1.0,  # image mosaic (probability)
    "mixup": 0.0,  # image mixup (probability)
    "copy_paste": 0.0,  # segment copy-paste (probability)
}

args_export = {
    "format": {
        "torchscript": 1,  # TorchScript
        "onnx": 1,  # ONNX
        "openvino": 0,  # OpenVINO
        "engine": 0,  # TensorRT
        "coreml": 0,  # CoreML
        "saved_model": 0,  # TensorFlow SavedModel
        "pb": 0,  # TensorFlow GraphDef
        "tflite": 0,  # TensorFlow Lite
        "edgetpu": 0,  # TensorFlow Edge TPU
        "tfjs": 0,  # TensorFlow.js
        "paddle": 0,  # PaddlePaddle
    },
    "params": {
        "keras": False,
        "optimize": False,
        "half": True,
        "int8": False,
        "dynamic": False,
        "simplify": False,
        "opset": None,
        "workspace": 4,
        "nms": False
    }
}

args_logging = {
    "project": "Debug/yolov8",
    "name": "training-yolo",
}

args_task = {
    "model_name": "yolov8n",
    "model_latest_id": ""
    # "pretrained": "(model_id)"
}

args_data = {
    "cvat": {
        "task_ids_train": [48],
        "task_ids_test": [],
    },
    "label_studio": {
        "project_id_train": None,
        "project_id_test": None,
    },
    "s3": {"s3_uri_dir_train": None, "s3_uri_dir_test": None},
    "params": {
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "test_ratio": None,
    },
    "class_exclude": None,
    # "attributes_exclude": {"maturity_truth": "background"},
    "attributes_exclude": None,
    "area_segment_min": 0,
}


args_train = {
    "augment": True,            
    "epochs": 1000,              # number of epochs to train for
    "patience": 0,             # epochs to wait for no observable improvement for early stopping of training
    "batch": 2,                # number of images per batch (-1 for AutoBatch)
    "imgsz": 640,               # size of input images as integer or w,h
    "save": True,               # save train checkpoints and predict results
    "save_period": -1,          # Save checkpoint every x epochs (disabled if < 1)
    "cache": True,             # True/ram, disk or False. Use cache for data loading
    "device": None,             # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    "workers": 8,               # number of worker threads for data loading (per RANK if DDP)
    "project": None,            # project name
    "name": None,               # experiment name
    "exist_ok": True,          # whether to overwrite existing experiment
    "pretrained": True,        # whether to use a pretrained model
    "optimizer": "auto",        # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    "verbose": True,           # whether to print verbose output
    "seed": 0,                  # random seed for reproducibility
    "deterministic": True,      # whether to enable deterministic mode
    "single_cls": False,        # train multi-class data as single-class
    "rect": False,              # rectangular training with each batch collated for minimum padding
    "cos_lr": False,            # use cosine learning rate scheduler
    "close_mosaic": 0,          # (int) disable mosaic augmentation for final epochs
    "resume": False,            # resume training from last checkpoint
    "amp": True,                # Automatic Mixed Precision (AMP) training, choices=[True, False]
    "fraction": 0.9,            # dataset fraction to train on (default is 1.0, all images in train set)
    "profile": False,           # profile ONNX and TensorRT speeds during training for loggers
    "lr0": 0.001,                # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.0001,                # final learning rate (lr0 * lrf)
    "momentum": 0.937,          # SGD momentum/Adam beta1
    "weight_decay": 0.0005,     # optimizer weight decay 5e-4
    "warmup_epochs": 3.0,       # warmup epochs (fractions ok)
    "warmup_momentum": 0.8,     # warmup initial momentum
    "warmup_bias_lr": 0.1,      # warmup initial bias lr
    "box": 7.5,                 # box loss gain
    "cls": 0.5,                 # cls loss gain (scale with pixels)
    "dfl": 1.5,                 # dfl loss gain
    "pose": 12.0,               # pose loss gain (pose-only)
    "kobj": 2.0,                # keypoint obj loss gain (pose-only)
    "label_smoothing": 0.0,     # label smoothing (fraction)
    "nbs": 64,                  # nominal batch size
    "overlap_mask": True,       # masks should overlap during training (segment train only)
    "mask_ratio": 4,            # mask downsample ratio (segment train only)
    "dropout": 0.0,             # use dropout regularization (classify train only)
    "val": True,               # validate/test during training
}

args_val = {
    "batch": 1,            # number of images per batch (-1 for AutoBatch)
    "save_json": False,     # save results to JSON file
    "save_hybrid": False,   # save hybrid version of labels (labels + additional predictions)
    "conf": 0.35,          # object confidence threshold for detection
    "iou": 0.6,             # intersection over union (IoU) threshold for NMS
    "max_det": 100,        # maximum number of detections per image
    "half": True,           # use half precision (FP16)
    "device": 0,         # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    "dnn": False,           # use OpenCV DNN for ONNX inference
    "plots": True,         # show plots during training
    "rect": False,          # rectangular val with each batch collated for minimum padding
    "split": "val",         # dataset split to use for validation, i.e. 'val', 'test' or 'train'
}