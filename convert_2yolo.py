import os
import json


def remove_all_extensions(filename):
    """Remove all extensions from a filename."""
    base, ext = os.path.splitext(filename)
    while ext:
        filename = base
        base, ext = os.path.splitext(filename)
    return filename


def convert_annotation(json_file, classes_mapping):
    with open(json_file, "r") as f:
        data = json.load(f)
    img_width = data["size"]["width"]
    img_height = data["size"]["height"]
    yolo_lines = []
    for obj in data.get("objects", []):
        original_class_id = obj.get("classId")
        # Map original class id to sequential index (0 to 5)
        if original_class_id not in classes_mapping:
            print(
                f"Warning: Class id {original_class_id} not found in mapping, skipping object."
            )
            continue
        class_idx = classes_mapping[original_class_id]

        # Get bounding box coordinates: first point is top-left, second is bottom-right
        x_min, y_min = obj["points"]["exterior"][0]
        x_max, y_max = obj["points"]["exterior"][1]

        # Calculate normalized center coordinates and dimensions
        center_x = ((x_min + x_max) / 2.0) / img_width
        center_y = ((y_min + y_max) / 2.0) / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height

        yolo_lines.append(
            f"{class_idx} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return yolo_lines


def process_annotations(input_dir, output_dir, classes_mapping):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            yolo_lines = convert_annotation(json_path, classes_mapping)
            # Remove all extensions from the filename
            base_name = remove_all_extensions(filename)
            txt_path = os.path.join(output_dir, base_name + ".txt")
            with open(txt_path, "w") as out_file:
                out_file.write("\n".join(yolo_lines))


# Define your mapping: original class id -> new sequential index
classes_mapping = {
    6488141: 0,  # bus
    6488139: 1,  # car
    6488143: 2,  # motorbike
    6488140: 3,  # threewheel
    6488142: 4,  # truck
    6488144: 5,  # van
}

# Process training annotations
train_ann_dir = "dataset/vehicle_dataset/train/ann"
train_labels_dir = "dataset/vehicle_dataset/train/labels"
process_annotations(train_ann_dir, train_labels_dir, classes_mapping)

# Process validation annotations
valid_ann_dir = "dataset/vehicle_dataset/valid/ann"
valid_labels_dir = "dataset/vehicle_dataset/valid/labels"
process_annotations(valid_ann_dir, valid_labels_dir, classes_mapping)
