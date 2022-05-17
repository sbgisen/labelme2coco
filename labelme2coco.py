import argparse
import glob
import json
import os
import pathlib

import numpy as np
import PIL.Image
from labelme import utils


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json", config_path=""):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.config_path = config_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                try:
                    data = json.load(fp)
                    self.images.append(self.image(data, num))
                    for shapes in data["shapes"]:
                        label = shapes["label"]
                        if label not in self.label:
                            self.label.append(label)
                        points = shapes["points"]
                        self.annotations.append(
                            self.annotation(points, label, num))
                        self.annID += 1
                except:
                    pass

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(
                annotation["category_id"])

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories) + 1
        category["name"] = label
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = label  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)
        parent = pathlib.Path(self.save_json_path).parent
        image_paths = [str(path.relative_to(parent))
                       for path in (parent/'train_data').glob('*.jpg')]
        # dataset_path (e.g.) custom_dataset/robocup_dataset
        dataset_path = str(parent.relative_to(parent.parent.parent))
        with open(parent/'train.txt', 'w') as f:
            f.write('\n'.join(image_paths))
        with open(parent/'classes.txt', 'w') as f:
            f.write('\n'.join(self.label))

        if self.config_path:
            context = []
            with open(self.config_path, 'r') as f:
                is_remove = False
                for line in f.readlines():
                    if line == f'# -----start {parent.name}-----\n':
                        is_remove = True
                    elif line == f'# -----end {parent.name}-----\n':
                        is_remove = False
                    elif not is_remove:
                        context.append(line)

            prefix = f'# -----start {parent.name}-----\n'
            suffix = f'# -----end {parent.name}-----\n'
            classes_str = '{}_CLASSES = ('.format(parent.name)

            for l in self.label:
                classes_str += f"'{l}',\n"
            classes_str += ')\n'
            dict_str = """
                'name': '{0} dataset',

                # Training images and annotations
                'train_images': './{1}/',
                'train_info':   './{1}/instances.json',

                # Validation images and annotations.
                'valid_images': './{1}/',
                'valid_info':   './{1}/instances.json',

                # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
                'has_gt': True,

                # A list of names for each of you classes.
                'class_names': {0}_CLASSES,

                # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
                # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
                # If not specified, this just assumes category ids start at 1 and increase sequentially.
                'label_map': None
            """.format(parent.name, dataset_path)
            dataset_str = '{}_dataset = dataset_base.copy('.format(
                parent.name)+'{'+dict_str+'})\n'

            dict_str = """
                'name': 'yolact_plus_{0}',

                # Dataset stuff
                'dataset': {0}_dataset,
                'num_classes': len({0}_dataset.class_names) + 1,
            """.format(parent.name)
            config_str = 'yolact_plus_{}_config = yolact_plus_resnet50_config.copy('.format(
                parent.name)+'{'+dict_str+'})\n'

            with open(self.config_path, 'w') as f:
                f.write(''.join(context)+prefix+classes_str +
                        dataset_str+config_str+suffix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--config",
        help="Directory to yolact config path.",
        default=""
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    labelme2coco(labelme_json, args.output, args.config)
