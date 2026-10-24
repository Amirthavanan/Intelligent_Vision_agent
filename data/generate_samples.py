import os
import json
import urllib.request
from io import BytesIO
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data")
SAMPLES_DIR = DATA_DIR / "sample_images"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_URLS = {
    "sample_street_traffic.jpg": "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?w=800&q=80",
    "sample_office_desk.jpg": "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=800&q=80",
    "sample_living_room.jpg": "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?w=800&q=80",
}


def download_or_verify_sample_images():
    for filename, url in SAMPLE_URLS.items():
        save_path = SAMPLES_DIR / filename
        if not save_path.exists() or save_path.stat().st_size < 1000:
            print(f"Downloading verified benchmark sample: {filename}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=15).read()
                im = Image.open(BytesIO(data)).convert("RGB")
                im = im.resize((800, 600), Image.Resampling.LANCZOS)
                im.save(save_path, "JPEG", quality=95)
                print(f"Saved {save_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"Sample image already present: {save_path}")


def generate_all_samples():
    download_or_verify_sample_images()

    gt = {
        "sample_street_traffic.jpg": [
            {"label": "car", "bbox": [23, 218, 760, 479]},
            {"label": "person", "bbox": [48, 139, 109, 278]},
            {"label": "person", "bbox": [9, 142, 54, 380]},
            {"label": "person", "bbox": [727, 233, 753, 332]},
        ],
        "sample_office_desk.jpg": [
            {"label": "laptop", "bbox": [110, 204, 564, 558]},
            {"label": "cup", "bbox": [36, 298, 114, 362]},
            {"label": "potted plant", "bbox": [196, 233, 246, 320]},
            {"label": "chair", "bbox": [25, 265, 71, 322]},
            {"label": "chair", "bbox": [0, 420, 163, 599]},
            {"label": "cell phone", "bbox": [79, 406, 221, 442]},
        ],
        "sample_living_room.jpg": [
            {"label": "potted plant", "bbox": [693, 212, 799, 450]},
            {"label": "couch", "bbox": [67, 288, 775, 436]},
            {"label": "potted plant", "bbox": [2, 264, 158, 410]},
            {"label": "vase", "bbox": [765, 353, 800, 438]},
        ],
    }

    gt_file = DATA_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Generated ground truth annotations at: {gt_file}")


if __name__ == "__main__":
    generate_all_samples()
