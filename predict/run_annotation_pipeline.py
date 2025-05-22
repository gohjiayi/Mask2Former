# run_annotation_pipeline.py

# fmt: off
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from detectron2.data.detection_utils import read_image
import torch
from torch.utils.data import Dataset, DataLoader
import time
import xml.etree.ElementTree as ET

from predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config

from cvat_utils import load_cvat_xml, get_image_elements, append_masks_to_image_element, write_xml

import mask2former.data.datasets.register_safetybarrier_instance

class ImageDataset(Dataset):
    def __init__(self, image_paths, image_elements):
        self.image_paths = image_paths
        self.image_elements = image_elements

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_el = self.image_elements[idx]
        img = read_image(str(img_path), format="BGR")
        return {
            'image': img,
            'element': img_el,
            'path': img_path
        }

def binary_mask_to_uncompressed_rle(binary_mask):
    """
    Efficient CVAT-compatible RLE encoder using NumPy vectorization.
    Assumes binary_mask is a 2D numpy array with values 0 or 1.
    Calculates RLE only for the bounding box region of the mask.
    """
    try:
        binary_mask = binary_mask.astype(np.uint8)
        
        # Find the bounding box coordinates
        y_indices, x_indices = np.nonzero(binary_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
            
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Extract the bounding box region
        bbox_mask = binary_mask[y1:y2+1, x1:x2+1]
        flat = bbox_mask.flatten(order='C')

        if flat[0] == 1:
            flat = np.insert(flat, 0, 0)

        # Find where values change
        diffs = np.diff(flat)
        change_indices = np.where(diffs != 0)[0] + 1
        boundaries = np.concatenate([[0], change_indices, [len(flat)]])
        rle = np.diff(boundaries)

        return ', '.join(str(x) for x in rle)

    except Exception as e:
        print(f"[RLE Conversion Error] {e}")
        return None


def setup_cfg(config_file, opts):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set device to CUDA if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg

def process_batch(batch, demo, vis_dir, pred_dir):
    # Extract images and metadata
    images = [item['image'] for item in batch]
    elements = [item['element'] for item in batch]
    paths = [item['path'] for item in batch]

    # Process batch of images
    predictions, visualized_outputs = demo.run_on_batch(images)

    # Process each prediction
    for pred, vis_output, img_el, img_path in zip(predictions, visualized_outputs, elements, paths):
        image_name = Path(img_path).name
        vis_path = vis_dir / image_name
        pred_path = pred_dir / (Path(image_name).stem + ".json")

        # Skip if already processed
        if vis_path.exists():
            continue

        # Save visualization
        vis_output.save(str(vis_path))

        # Process predictions
        inst = pred["instances"].to("cpu")
        masks = inst.pred_masks.numpy()
        boxes = inst.pred_boxes.tensor.numpy()
        classes = inst.pred_classes.tolist()
        scores = inst.scores.tolist()
        class_names = [demo.metadata.thing_classes[i] for i in classes]

        # Filter predictions based on confidence threshold
        valid_indices = [i for i, score in enumerate(scores) if score >= demo.confidence_threshold]

        image_preds = []
        for i in valid_indices:
            mask = masks[i]
            # Get RLE encoding
            rle = binary_mask_to_uncompressed_rle(mask)
            if rle is None:
                print(f"Warning: Failed to convert mask to RLE for {image_name}")
                continue

            # Calculate bounding box from mask
            # Find the non-zero coordinates in the mask
            y_indices, x_indices = np.nonzero(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Warning: Empty mask found for {image_name}")
                continue  # Skip empty masks

            # Calculate bounding box
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            mask_width = x2 - x1 + 1
            mask_height = y2 - y1 + 1

            # Get image dimensions from mask
            img_height, img_width = mask.shape

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            mask_width = min(mask_width, img_width - x1)
            mask_height = min(mask_height, img_height - y1)

            image_preds.append({
                "label": class_names[i],
                "rle": rle,
                "left": int(x1),
                "top": int(y1),
                "width": int(mask_width),
                "height": int(mask_height)
            })

        if not image_preds:
            print(f"Warning: No valid predictions for {image_name}")
            continue

        # Save prediction JSON
        with open(pred_path, "w") as f:
            json.dump(image_preds, f)

        # Update XML
        append_masks_to_image_element(img_el, image_preds)

def load_existing_predictions(pred_dir):
    """Load all existing predictions from JSON files."""
    predictions = {}
    for json_file in pred_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                preds = json.load(f)
                if preds:  # Only add if there are predictions
                    predictions[json_file.stem] = preds
                    print(f"Loaded {len(preds)} predictions from {json_file.name}")
                else:
                    print(f"Warning: Empty predictions in {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {str(e)}")
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing images/ and annotations.xml")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs: XML, predictions, and visuals")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Optional config opts")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading")
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    image_dir = input_dir / "images"
    annotation_path = input_dir / "annotations.xml"
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visuals"
    pred_dir = output_dir / "predictions"
    final_xml_path = output_dir / "annotations.xml"

    # Check if output directory or subfolders already contain files
    output_exists = (
        final_xml_path.exists() or
        (vis_dir.exists() and any(vis_dir.iterdir())) or
        (pred_dir.exists() and any(pred_dir.iterdir()))
    )

    if output_exists:
        print(f"\n⚠️ Output already exists in '{output_dir}'.")
        user_input = input("Do you want to resume from existing outputs? [Y/n]: ").strip().lower()

        if user_input == 'n':
            print("🗑️  Overwriting existing outputs...")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
            pred_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("🔄 Resuming from existing outputs...")

    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup predictor
    setup_logger()
    cfg = setup_cfg(args.config_file, args.opts)
    demo = VisualizationDemo(cfg, confidence_threshold=args.confidence_threshold)

    # Load annotation XML
    tree, root = load_cvat_xml(annotation_path)
    image_elements = get_image_elements(root)

    # Load existing predictions if resuming
    existing_predictions = {}
    if output_exists and user_input != 'n':
        existing_predictions = load_existing_predictions(pred_dir)
        print(f"Loaded predictions for {len(existing_predictions)} images")

    # Filter out already processed images
    image_paths = []
    filtered_elements = []
    for img_el in image_elements:
        image_name = img_el.attrib["name"]
        image_stem = Path(image_name).stem
        vis_path = vis_dir / image_name
        pred_path = pred_dir / (image_stem + ".json")

        # If we have both visualization and prediction, use existing prediction
        if vis_path.exists() and pred_path.exists():
            if image_stem in existing_predictions:
                preds = existing_predictions[image_stem]
                if preds:  # Only append if there are predictions
                    print(f"Adding {len(preds)} masks to {image_name}")
                    append_masks_to_image_element(img_el, preds)
                else:
                    print(f"Warning: No predictions found for {image_name}")
            continue

        image_paths.append(image_dir / image_name)
        filtered_elements.append(img_el)

    if not image_paths:
        print("All images have been processed!")
        # Write final XML with all predictions
        write_xml(tree, final_xml_path)
        return

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, filtered_elements)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda x: x  # Keep as list of dicts
    )

    # Process batches
    start_time = time.time()
    for batch in tqdm(dataloader, desc="Processing batches"):
        process_batch(batch, demo, vis_dir, pred_dir)

    # After processing all batches, verify and update XML
    print("\nVerifying predictions and updating XML...")
    for img_el in image_elements:
        image_name = img_el.attrib["name"]
        image_stem = Path(image_name).stem
        pred_path = pred_dir / (image_stem + ".json")
        
        if pred_path.exists():
            try:
                with open(pred_path, "r") as f:
                    preds = json.load(f)
                    if preds:
                        print(f"Adding {len(preds)} masks to {image_name}")
                        append_masks_to_image_element(img_el, preds)
                    else:
                        print(f"Warning: No predictions found in {pred_path}")
            except Exception as e:
                print(f"Error loading predictions for {image_name}: {str(e)}")

    # Write final XML after all processing is complete
    write_xml(tree, final_xml_path)

    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
