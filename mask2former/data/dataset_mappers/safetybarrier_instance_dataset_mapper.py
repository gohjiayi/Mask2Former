import copy
import torch

from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper \
    import COCOInstanceNewBaselineDatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class SafetyBarrierInstanceDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.is_train = is_train

        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT

        # Define LSJ-style augmentations manually
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip()
        ])

        self.keypoint_hflip_indices = (
            cfg.DATASETS.KEYPOINT_HFLIP_INDICES if hasattr(cfg.DATASETS, "KEYPOINT_HFLIP_INDICES") else None
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # Apply transforms to annotations
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image.shape[:2],
                keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]

        # Create Instances using bitmask mode
        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.mask_format
        )
        instances.gt_masks = instances.gt_masks.tensor

        # Convert to CHW tensor (Detectron2 expects this)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = instances
        return dataset_dict
