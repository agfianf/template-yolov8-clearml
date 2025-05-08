import json
import os
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from src.schema.coco import Coco as CocoSchema
from glob import glob
from uuid import uuid4

image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'ico', 'webp', 'svg']

def count_files_in_directory(path, extensions=None):
    """
    Menghitung jumlah file dalam folder secara rekursif.

    Parameters:
    path (str): Path direktori utama yang akan dihitung.
    extensions (list of str, optional): Daftar ekstensi file yang ingin digunakan sebagai filter.
        Jika tidak ditentukan (None), maka semua jenis file akan dihitung.

    Returns:
    int: Jumlah file dalam folder yang sesuai dengan filter ekstensi jika ditentukan.

    Example:
    >>> folder_path = "/path/to/your/folder"
    >>> extensions_to_count = [".txt", ".csv", ".jpg"]
    >>> total_files = count_files_in_directory(folder_path, extensions_to_count)
    >>> print(f"Total files: {total_files}")
    """
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if extensions is None or file.endswith(tuple(extensions)):
                count += 1
    return count

class Coco2Yolo:
    def __init__(self, src_dir, output_dir='./yolov8-dataset'):
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.src_img_dir = os.path.join(self.src_dir, 'images')
        self.src_lbl_filepath = os.path.join(self.src_dir, 'annotations', 'instances_default.json')
        # print(self.__dict__)

    @staticmethod
    def __min_index(arr1, arr2):
        """Find a pair of indexes with the shortest distance. 
        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            a pair of indexes(tuple).
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    @staticmethod
    def __merge_multi_segment(segments):
        """Merge multi segments to one list.
        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all 
        segments into one.
        Args:
            segments(List(List)): original segmentations in coco's json file.
                like [segmentation1, segmentation2,...], 
                each segmentation is a list of coordinates.
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = Coco2Yolo.__min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        for k in range(2):
            # forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s
    
    @staticmethod
    def convert_coco_to_yolo(json_path, use_segments=False, output_dir='./labels', exclude_class=[], attributes_excluded=None, area_segment_min=None):
        """
        Convert coco format to yolo format.
        Args:
            json_path: (str) path to coco json file.
            use_segments: (bool) whether to use segments to represent bbox.
            output_dir: (str) path to save yolo format labels.
        Return:
            list of labels.
        """
        if os.path.exists(output_dir):
            print(f"❌ [convert_coco_to_yolo] Remove {output_dir}")
            shutil.rmtree(output_dir)
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(json_path) as f:
            coco_d = json.load(f)

        coco = CocoSchema(**coco_d)

        images = coco.get_imageid_to_image()
        imgToAnns = coco.get_imageid_to_annotations(
                exclude_class=exclude_class,
                attributes_excluded=attributes_excluded,
                area_segment_min=area_segment_min
            )
        
        # write labels in txt file
        cat_id2name =coco.get_categoryid_to_namecat(exclude_class=exclude_class)

        for image_id, img_annotatins in tqdm(imgToAnns.items(), desc="Converting COCO to YOLO"):
            img = images[image_id]
            h, w, fp_image =img.height, img.width, img.file_name

            bboxes = []
            segments = []

            for ann in img_annotatins:
                if ann.iscrowd:
                    continue

                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann.bbox, dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue
                
                cls = ann.category_id - 1 # here because category_id starts from 1
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                
                if use_segments:
                    if len(ann.segmentation) > 1:
                        s = Coco2Yolo.__merge_multi_segment(ann.segmentation)
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann.segmentation for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write for each images
            filename = os.path.basename(fp_image)
            labelname = os.path.splitext(filename)[0] + '.txt'
            txt_file = os.path.join(output_dir, labelname)
            with open(txt_file, 'a') as file:
                for i in range(len(bboxes)):
                    try:
                        line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')
                    except Exception as e:
                        print("Error", e)
                        raise Exception(f"ERROR CONVERTER: {txt_file} idx:{i} use_segments={use_segments} -> {e} >> {bboxes}")

        return list(cat_id2name.values())
    
    def setup_directory(self):
        print("setup_directory runnnig")

        # create new output directry
        self.out_img_dir = os.path.join(self.output_dir, 'images')
        self.out_lbl_dir = os.path.join(self.output_dir, 'labels')
        Path(self.out_img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_lbl_dir).mkdir(parents=True, exist_ok=True)

        print(self.src_img_dir, os.path.exists(self.src_img_dir))
        # copy to new output directory with uuid name
        count_files = 0
        
        # tell len of images and labels
        total_images = count_files_in_directory(self.src_img_dir, extensions=image_extensions)
        total_labels = count_files_in_directory(self.src_lbl_yolo, extensions=[".txt"])
        print("is_match:", total_images==total_labels, "len images", total_images, "len labels", total_labels)
        
        count_no_annotations = 0
        
        for root, dirs, files in os.walk(self.src_img_dir):
            for filename in files:
                
                filename_only, ext = os.path.splitext(filename)
                new_filename_wo_ext = filename_only+'_' +str(uuid4().hex)
                new_filename_img = new_filename_wo_ext + ext
                new_filename_lbl = new_filename_wo_ext + '.txt'

                # images
                src_img_file = os.path.join(root, filename)
                dest_img_file = os.path.join(self.out_img_dir, new_filename_img)

                # labels
                src_lbl_file = os.path.join(self.src_lbl_yolo, filename_only+'.txt')
                dest_lbl_file = os.path.join(self.out_lbl_dir, new_filename_lbl)

                if os.path.exists(src_lbl_file):
                    shutil.copy2(src_img_file, dest_img_file)
                    shutil.copy2(src_lbl_file, dest_lbl_file)
                    count_files+=1
                else:
                    count_no_annotations+=1
        
        print("is_match",count_files==count_no_annotations, "count_files project:", count_files, "count_no_annotations project:", count_no_annotations)
        return count_files

    def convert(self, use_segments:bool, exclude_class=[], attributes_excluded=None, area_segment_min=None):
        print("Start Converting and Filtering COCO to YOLO")
        self.src_lbl_yolo = os.path.join(self.src_dir, "labels")
        list_categories = Coco2Yolo.convert_coco_to_yolo(
            json_path=self.src_lbl_filepath, 
            use_segments=use_segments,
            output_dir=self.src_lbl_yolo,
            exclude_class=exclude_class,
            attributes_excluded=attributes_excluded,
            area_segment_min=area_segment_min
        )
        print("Setup Directory: Manage Files to structure of YOLO")
        conut_data = self.setup_directory()
        print("Done Converting and Filtering COCO to YOLO")
        return self.output_dir, list_categories, conut_data
    
if __name__ == "__main__":
    print("start testing")
    ls_path_dir_projects = [
        "fyypp-numplate-no-aug",
        "IOTSmartCampus",
        "truckplate"
    ]
    ls_path_dir_projects = [os.path.join("tmp-cvat/Plate-Detector", path) for path in ls_path_dir_projects]
    if os.path.exists("./testing-debug-ds"):
        print("❌ Remove testing-debug-ds")
        shutil.rmtree("./testing-debug-ds")
    
    tmp_total_count = 0
    for project_dir in ls_path_dir_projects:
        converter = Coco2Yolo(src_dir=project_dir, output_dir="./testing-debug-ds")
        converter.src_lbl_yolo = os.path.join(converter.src_dir, "labels")
        converter.convert_coco_to_yolo(
            json_path=converter.src_lbl_filepath,
            use_segments=False,
            output_dir=converter.src_lbl_yolo,
            exclude_class=[],
            attributes_excluded=None,
            area_segment_min=None
        )
        tmp_total_count+= converter.setup_directory()
    
    countfiles_finel = len(os.listdir("./testing-debug-ds/images"))
    print("countfiles_finel", countfiles_finel, tmp_total_count, tmp_total_count==countfiles_finel)
