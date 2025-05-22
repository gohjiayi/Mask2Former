# cvat_utils.py

import xml.etree.ElementTree as ET
from pathlib import Path

def load_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return tree, root

def get_image_elements(root):
    return root.findall("image")

def append_masks_to_image_element(image_el, prediction_dict):
    """Append mask elements to an image element.
    
    Args:
        image_el: The image element to append masks to
        prediction_dict: List of dictionaries containing mask predictions
    """
    # Remove any existing mask elements to avoid duplicates
    for mask_el in image_el.findall("mask"):
        image_el.remove(mask_el)
        
    # Add new mask elements
    for pred in prediction_dict:
        mask_el = ET.SubElement(image_el, "mask")
        mask_el.set("label", pred["label"])
        mask_el.set("source", "semi-auto")
        mask_el.set("occluded", "0")
        mask_el.set("rle", pred["rle"])
        mask_el.set("left", str(pred["left"]))
        mask_el.set("top", str(pred["top"]))
        mask_el.set("width", str(pred["width"]))
        mask_el.set("height", str(pred["height"]))
        mask_el.set("z_order", "0")
        # Add empty text to ensure proper closing tag
        mask_el.text = " "

def write_xml(tree, output_path):
    """Write the XML tree to a file with proper structure.
    
    Args:
        tree: The XML tree to write
        output_path: Path where to write the XML file
    """
    # Get the original root
    original_root = tree.getroot()
    
    # Create new root element
    root = ET.Element("annotations")
    
    # Copy version
    version = ET.SubElement(root, "version")
    version.text = "1.1"
    
    # Copy meta section exactly as is
    meta = original_root.find("meta")
    if meta is not None:
        root.append(meta)
    
    # Get all image elements from the original tree
    image_elements = original_root.findall("image")
    
    # Copy all image elements with their masks
    for img_el in image_elements:
        # Create new image element with same attributes
        new_img_el = ET.SubElement(root, "image")
        for key, value in img_el.attrib.items():
            new_img_el.set(key, value)
        
        # Copy all mask elements
        masks = img_el.findall("mask")
        if not masks:
            print(f"Warning: No masks found for image {img_el.get('name')}")
            continue
            
        for mask_el in masks:
            new_mask_el = ET.SubElement(new_img_el, "mask")
            for key, value in mask_el.attrib.items():
                new_mask_el.set(key, value)
            # Add empty text to ensure proper closing tag
            new_mask_el.text = " "
    
    # Create new tree with the modified root
    new_tree = ET.ElementTree(root)
    
    # Write with proper XML declaration and encoding
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    # Verify the written file
    try:
        verification_tree = ET.parse(output_path)
        verification_root = verification_tree.getroot()
        
        # Verify masks are present
        total_masks = 0
        for img_el in verification_root.findall("image"):
            masks = img_el.findall("mask")
            if not masks:
                print(f"Warning: No masks found in output XML for image {img_el.get('name')}")
            else:
                total_masks += len(masks)
                print(f"Found {len(masks)} masks for image {img_el.get('name')}")
        
        print(f"\nTotal masks in XML: {total_masks}")
        if total_masks == 0:
            print("Warning: No masks were written to the XML file!")
                
    except ET.ParseError as e:
        print(f"Warning: XML validation failed: {str(e)}")
        # Try to fix common XML issues
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Replace any invalid XML characters
        content = content.replace('\x00', '')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
