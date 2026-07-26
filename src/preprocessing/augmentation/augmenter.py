import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from preprocessing.pipeline.helpers import logger

class ImageAugmenter:
    def augment_image(self, image_path: Path, output_dir: Path) -> list:
        """Applies spatial and color transformations, returning list of augmented file paths."""
        augmented_files = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            img = Image.open(image_path)
            base_name = image_path.stem
            ext = image_path.suffix
            
            # Save original first
            shutil_copy = output_dir / f"{base_name}_orig{ext}"
            img.save(shutil_copy)
            augmented_files.append(shutil_copy)

            # 1. Horizontal Flip
            h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            h_flip_path = output_dir / f"{base_name}_hflip{ext}"
            h_flip.save(h_flip_path)
            augmented_files.append(h_flip_path)
            
            # 2. Rotation 90
            rot_90 = img.rotate(90)
            rot_90_path = output_dir / f"{base_name}_rot90{ext}"
            rot_90.save(rot_90_path)
            augmented_files.append(rot_90_path)

            # 3. Brightness Adjustment
            enhancer = ImageEnhance.Brightness(img)
            bright = enhancer.enhance(1.2)
            bright_path = output_dir / f"{base_name}_bright{ext}"
            bright.save(bright_path)
            augmented_files.append(bright_path)

            # 4. Blur Filter
            blur = img.filter(ImageFilter.GaussianBlur(1.0))
            blur_path = output_dir / f"{base_name}_blur{ext}"
            blur.save(blur_path)
            augmented_files.append(blur_path)
            
            logger.info(f"Generated 4 augmentations for image: {image_path.name}")
        except Exception as e:
            logger.error(f"Augmentation failed for {image_path.name}: {e}")
            
        return augmented_files
