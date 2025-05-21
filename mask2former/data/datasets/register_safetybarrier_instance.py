from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

classes = ["rail_top", "rail_mid", "toeboard", "vertical_post"]  # keep order!

for split in ["train", "val"]:
    json_file = f"datasets/safetybarrier/annotations/instances_{split}.json"
    img_root  = f"datasets/safetybarrier/images/{split}"
    name      = f"safetybarrier_{split}"
    register_coco_instances(name, {}, json_file, img_root)
    MetadataCatalog.get(name).thing_classes = classes
