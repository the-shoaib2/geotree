import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Tuple
from preprocessing.pipeline.helpers import logger, ensure_dir

class ImageTiler:
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap

    def tile_image(self, tiff_path: Path, output_dir: Path) -> List[Path]:
        """Tiles a GeoTIFF image into overlapping patches, saving each non-empty patch."""
        tiled_paths = []
        ensure_dir(output_dir)
        
        try:
            with rasterio.open(tiff_path) as src:
                width = src.width
                height = src.height
                
                stride = self.tile_size - self.overlap
                
                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        # Calculate window coordinates
                        w_width = min(self.tile_size, width - x)
                        w_height = min(self.tile_size, height - y)
                        
                        if w_width < 128 or w_height < 128:
                            # Skip tiny edge tiles
                            continue
                            
                        window = Window(x, y, w_width, w_height)
                        data = src.read(window=window)
                        
                        # Filter out empty or mostly "no-data" black tiles
                        if data.max() == 0 or (data == 0).sum() / data.size > 0.8:
                            continue
                            
                        # Set window metadata
                        win_transform = rasterio.windows.transform(window, src.transform)
                        meta = src.meta.copy()
                        meta.update({
                            "height": w_height,
                            "width": w_width,
                            "transform": win_transform
                        })
                        
                        tile_name = f"{tiff_path.stem}_tile_{y}_{x}.tif"
                        tile_path = output_dir / tile_name
                        
                        with rasterio.open(tile_path, "w", **meta) as dst:
                            dst.write(data)
                            
                        tiled_paths.append(tile_path)
            logger.info(f"Tiled {tiff_path.name}: Generated {len(tiled_paths)} patches inside {output_dir.name}")
        except Exception as e:
            logger.error(f"Tiling failure for {tiff_path.name}: {e}")
            
        return tiled_paths
