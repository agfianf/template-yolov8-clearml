import os
import random
import shutil

from pathlib import Path

from tqdm import tqdm


def creating_yaml_file(source_dir, labels: list = None):
    """Creating data.yaml file for yolov8
    source_dir: Directory path where the images and labels are located
                expected classes.txt is located too.
    labels: (list) list of labels if classes.txt not provided
    """
    if labels is None:
        fp_label_txt = os.path.join(source_dir, "classes.txt")
        if not os.path.exists(fp_label_txt):
            print(f"yaml_path: {fp_label_txt}")
            raise Exception("[Creating YAML] classes.txt not found")
        with open(fp_label_txt) as f:
            labels = f.readlines()
            labels = [line.strip() for line in labels]

    out_fp_yaml = os.path.join(source_dir, "data.yaml")

    assert len(labels) > 0, "labels is empty"

    use_test = False
    ls_dir = ["train", "valid"]
    if os.path.exists(os.path.join(source_dir, "test")):
        ls_dir.append("test")

    for dir in ls_dir:
        dir_img = os.path.join(source_dir, dir, "images")
        dir_labels = os.path.join(source_dir, dir, "labels")

        # checkhing
        is_dir_exist = os.path.exists(dir_img) and os.path.exists(dir_labels)
        is_have_files = len(os.listdir(dir_img)) > 0 and len(os.listdir(dir_labels)) > 0
        is_exact_count = len(os.listdir(dir_img)) == len(os.listdir(dir_labels))

        is_passed = is_dir_exist and is_have_files and is_exact_count

        if dir == "test":
            if is_passed:
                use_test = True
        else:
            if not is_passed:
                print
                print(f"yaml_path: {out_fp_yaml}")
                print(f"""
                {dir_img}
                {dir_labels}
                {os.listdir(dir_img)}
                {os.listdir(dir_labels)}
                      
                is_dir_exist:{is_dir_exist}
                is_have_files:{is_have_files}
                is_exact_count:{is_exact_count}


                """)
                raise Exception(f"""[Creating YAML] {dir} folder is not valid. 
                                is_dir_exist:{is_dir_exist}
                                is_have_files:{is_have_files}
                                is_exact_count:{is_exact_count}
                                """)

    with open(out_fp_yaml, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        if use_test:
            f.write("test: test/images\n\n")
        f.write(f"nc: {len(labels)}\n")
        f.write(f"names: {[name for name in labels]}")
    return out_fp_yaml


def split_folder_yolo(source_dir, train_ratio=0.8, valid_ratio=0.2, test_ratio=None):
    """Split folder yolo into train, valid, test (optional)
    source_dir: Directory path where the images and labels are located.
    train_ratio: Ratio of the dataset to be used for training (default: 0.7).
    valid_ratio: Ratio of the dataset to be used for validation (default: 0.15).
    test_ratio: Ratio of the dataset to be used for testing (default: 0.15).
    """
    # Create train, valid, and test directories
    src_dir_images = os.path.join(source_dir, "images")
    src_dir_labels = os.path.join(source_dir, "labels")

    if not os.path.exists(src_dir_images) and not os.path.exists(src_dir_labels):
        raise Exception("[Splitting] source_dir: images and labels directory not found")

    dst_train_dir = os.path.join(source_dir, "train")
    dst_valid_dir = os.path.join(source_dir, "valid")
    ls_dirs = [dst_train_dir, dst_valid_dir]

    if test_ratio:
        dst_test_dir = os.path.join(source_dir, "test")
        ls_dirs.append(dst_test_dir)

    for dir in ls_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        image_dir = os.path.join(dir, "images")
        label_dir = os.path.join(dir, "labels")
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        Path(label_dir).mkdir(parents=True, exist_ok=True)

    # Get the list of image files in the source directory
    ls_images_labels = list()
    for root, dirs, files in os.walk(src_dir_images):
        for filename_image in files:
            filename_only, ext = os.path.splitext(filename_image)
            filename_label = filename_only + ".txt"
            fp_image = os.path.join(src_dir_images, filename_image)
            fp_label = os.path.join(src_dir_labels, filename_label)
            if os.path.exists(fp_label) and os.path.exists(fp_image):
                ls_images_labels.append((fp_image, fp_label))

    # Randomly shuffle the image files
    if len(ls_images_labels) == 0:
        raise Exception("[Splitting] Error. No images found in the source directory!")

    random.seed(42)  # Ensure reproducibility across runs
    random.shuffle(ls_images_labels)

    # Calculate the number of images for each set
    total_files = len(ls_images_labels)
    num_train = int(total_files * train_ratio)
    if test_ratio:
        num_valid = int(total_files * valid_ratio)
    else:
        num_valid = int(total_files * (1 - train_ratio))

    ls_train = ls_images_labels[:num_train]
    ls_valid = ls_images_labels[num_train : num_train + num_valid]
    ls_files = [ls_train, ls_valid]

    if test_ratio:
        ls_test = ls_images_labels[num_train + num_valid :]
        ls_files.append(ls_test)

    # Copy images and labels to the corresponding sets
    for list_files, dst_dir in zip(ls_files, ls_dirs, strict=False):
        for src_fp_image, src_fp_label in tqdm(list_files, desc="Splitting files"):
            dst_fp_image = os.path.join(dst_dir, "images", os.path.basename(src_fp_image))
            dst_fp_label = os.path.join(dst_dir, "labels", os.path.basename(src_fp_label))
            shutil.move(src_fp_image, dst_fp_image)
            shutil.move(src_fp_label, dst_fp_label)

    shutil.rmtree(src_dir_images)
    shutil.rmtree(src_dir_labels)


def creating_classes_txt(dataset_dir, label_names):
    with open(os.path.join(dataset_dir, "classes.txt"), "w") as file:
        for cls_name in label_names:
            file.write(cls_name + "\n")


def setup_dataset(
    dataset_dir,
    label_names,
    train_ratio=0.8,
    valid_ratio=0.2,
    test_ratio=None,
    dataset_test=None,
):
    creating_classes_txt(dataset_dir, label_names)

    split_folder_yolo(
        source_dir=dataset_dir,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    if dataset_test:
        if os.path.exists(f"{dataset_dir}/test"):
            shutil.rmtree(f"{dataset_dir}/test")
        Path(f"{dataset_dir}/test").mkdir(parents=True, exist_ok=True)

        shutil.move(dataset_test + "/images", f"{dataset_dir}/test")
        shutil.move(dataset_test + "/labels", f"{dataset_dir}/test")

    creating_yaml_file(dataset_dir)

    if os.path.exists(f"{dataset_dir}/test/labels.cache"):
        os.remove(f"{dataset_dir}/test/labels.cache")
    if os.path.exists(f"{dataset_dir}/train/labels.cache"):
        os.remove(f"{dataset_dir}/train/labels.cache")
    if os.path.exists(f"{dataset_dir}/valid/labels.cache"):
        os.remove(f"{dataset_dir}/valid/labels.cache")

    return dataset_dir


def cleanup_cache(dataset_dir):
    for section in ["train", "valid", "test"]:
        path_cache = os.path.join(dataset_dir, section, "labels.cache")
        if os.path.exists(path_cache):
            print(f"Removing {path_cache}")
            os.remove(path_cache)


if __name__ == "__main__":
    # Test code
    split_folder_yolo("")
