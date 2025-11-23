from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_track_info, extract_kart_objects
import json


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    captions = []

    # get all karts
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if len(karts) == 0:
        return ["There are no karts in the scene."]

    # ego kart
    ego = [k for k in karts if k["is_center_kart"]]
    ego_name = ego[0]["kart_name"] if ego else "the ego car"

    # 1. Ego caption
    captions.append(f"{ego_name} is the ego car.")

    # 2. Count caption
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3. Track caption
    track = extract_track_info(info_path)
    captions.append(f"The track is {track}.")

    # 4. Relative position captions
    if ego:
        ex, ey = ego[0]["center"]

        for k in karts:
            if k["is_center_kart"]:
                continue

            name = k["kart_name"]
            x, y = k["center"]

            left_or_right = "left" if x < ex else "right"
            front_or_back = "in front" if y < ey else "behind"

            captions.append(f"{name} is {front_or_back} of the ego car.")
            captions.append(f"{name} is to the {left_or_right} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def build_caption_file(info_dir="data/train", output_name="train_captions.json"):
    info_dir = Path(info_dir)
    out_path = info_dir / output_name

    all_pairs = []

    info_files = sorted(info_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files. Generating captions...")

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")

        # 4 views per frame
        for view_index in range(4):
            img_list = list(info_file.parent.glob(f"{base}_{view_index:02d}_im.jpg"))
            if not img_list:
                continue

            img_file = img_list[0].name   # just "00000_00_im.jpg"

            # generate captions
            caps = generate_caption(str(info_file), view_index)

            for c in caps:
                all_pairs.append({
                    "image_file": f"train/{img_file}",
                    "caption": c
                })

    print(f"Generated {len(all_pairs)} caption pairs. Saving to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print("Done!")


def main():
    fire.Fire({"check": check_caption, "build": build_caption_file,})


if __name__ == "__main__":
    main()
