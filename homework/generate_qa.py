import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path, "r") as f:
        info = json.load(f)

    dets_per_view = info["detections"]
    kart_names = info["karts"]

    if view_index >= len(dets_per_view):
        return []

    dets = dets_per_view[view_index]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []

    for det in dets:
        class_id, track_id, x1, y1, x2, y2 = det

        if class_id != 1:  # only kart detections
            continue

        # Scale bounding box
        xs1 = x1 * scale_x
        ys1 = y1 * scale_y
        xs2 = x2 * scale_x
        ys2 = y2 * scale_y

        if (xs2 - xs1) < min_box_size or (ys2 - ys1) < min_box_size:
            continue

        cx = (xs1 + xs2) / 2
        cy = (ys1 + ys2) / 2

        # Convert track_id → actual kart name
        # track_id indexes into info["karts"]
        kart_name = kart_names[int(track_id)]

        karts.append({
            "instance_id": int(track_id),
            "kart_name": kart_name,
            "center": (cx, cy),
            "is_center_kart": False
        })

    # Identify ego car (closest to center of resized image)
    if karts:
        img_cx = img_width / 2
        img_cy = img_height / 2
        ego = min(karts, key=lambda k: (k["center"][0] - img_cx)**2 + (k["center"][1] - img_cy)**2)
        ego["is_center_kart"] = True

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path, "r") as f:
        info = json.load(f)

    return info.get("track", "unknown track")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego = [k for k in karts if k["is_center_kart"]][0]
    ego_cx, ego_cy = ego["center"]

    track_name = extract_track_info(info_path)

    # -------------------------------------------------------
    # (1) Ego kart question
    # -------------------------------------------------------
    qa.append({
        "question": "Which kart is the ego car?",
        "answer": ego["kart_name"]
    })

    # -------------------------------------------------------
    # (2) Total kart count
    # -------------------------------------------------------
    qa.append({
        "question": "How many karts are in the scene?",
        "answer": str(len(karts))
    })

    # -------------------------------------------------------
    # (3) Track name
    # -------------------------------------------------------
    qa.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # -------------------------------------------------------
    # (4) For each other kart: left/right/front/behind
    # -------------------------------------------------------
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    for k in karts:
        if k["is_center_kart"]:
            continue

        name = k["kart_name"]
        cx, cy = k["center"]

        # left vs right
        lr = "left" if cx < ego_cx else "right"
        if lr == "left":
            left_count += 1
        else:
            right_count += 1

        # front vs behind (smaller y = more in front)
        fb = "in front of" if cy < ego_cy else "behind"
        if fb == "in front of":
            front_count += 1
        else:
            behind_count += 1

        # Q1: left or right?
        qa.append({
            "question": f"Is {name} to the left or right of the ego car?",
            "answer": lr
        })

        # Q2: in front or behind?
        qa.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": "in front" if fb == "in front of" else "behind"
        })

        # Q3: combined relation
        qa.append({
            "question": f"Where is {name} relative to the ego car?",
            "answer": f"{lr} and {fb}"
        })

    # -------------------------------------------------------
    # (5) Counting questions
    # -------------------------------------------------------
    qa.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count)
    })
    qa.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count)
    })
    qa.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count)
    })
    qa.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind_count)
    })

    return qa


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def build_training_qa_file(info_dir: str = "data/train", output_name: str = "train_qa_pairs1.json"):
    """
    Generate QA pairs for all frames in a directory and save them to a JSON file.
    
    Args:
        info_dir: Directory containing *_info.json files
        output_name: Output JSON filename (saved inside info_dir)
    """
    info_dir = Path(info_dir)
    out_path = info_dir / output_name
    
    all_pairs = []
    info_files = sorted(info_dir.glob("*_info.json"))

    print(f"Found {len(info_files)} info files. Generating QA pairs...")

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")  # e.g. 00000
        # there are 4 views: 00, 01, 02, 03
        for view_index in range(4):
            # Check if the image exists
            img = list(info_file.parent.glob(f"{base}_{view_index:02d}_im.jpg"))
            if len(img) == 0:
                # no image for this view → skip safely
                continue
            
            try:
                qa = generate_qa_pairs(str(info_file), view_index)
                for q in qa:
                    # add image filename reference
                    q["image_file"] = f"train/{base}_{view_index:02d}_im.jpg"
                    all_pairs.append(q)
            except Exception as e:
                print(f"Error generating QA for {info_file} view {view_index}: {e}")

    print(f"Generated {len(all_pairs)} QA pairs. Saving to {out_path}...")
    
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print("Done!")


def main():
    fire.Fire({"check": check_qa_pairs,  "build": build_training_qa_file,  })


if __name__ == "__main__":
    main()
