import os
import cv2
from pathlib import Path

def crop_every_image(img_path: str):
    
    crop_img_path = Path("data/crop_images")
    crop_img_path.mkdir(parents=True, exist_ok=True)

    for img in img_path:
        if img.exists():
            filename = img.name
            img = cv2.imread(str(img))
            x1 = 325
            y1 = 201
            x2 = 556
            y2 = 413

            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]

            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            cropped_img = img[y1:y2, x1:x2]
            save_path = crop_img_path / filename
            cv2.imwrite(str(save_path), cropped_img)
            print(f"Cropped image saved to: {save_path}")
    return 

def main():
    raw_img_path = Path("data/test_images")
    img_files = list(raw_img_path.glob("*.jpg"))
    crop_every_image(img_files)

if __name__ == "__main__":
    main()