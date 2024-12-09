import numpy as np
import os
import rasterio
import PIL
import torch
from RealESRGAN import RealESRGAN  ## lib requirements hadnt been freezed
                                   ## require huggingface_hub==v0.11.1


def get_tif_images_path_list(base_path: str) -> list:
    tif_paths = [f.path 
                 for f in os.scandir(base_path) 
                 if f.is_file() and os.path.splitext(f.name)[1] == ".tif" 
                 ]
    
    return tif_paths


def read_landsat_band_from_tif_file(image_path: str, band_idx: int):
    with rasterio.open(image_path) as src:
        return src.read(band_idx)


def normalize_band(band):
    assert band.any(), """band needs non-na value. band.any() returned False"""
    # min-max scaling
    return (band - band.min()) / (band.max() - band.min())

def process_single_image(image_path: str, bands: list):
    _image_arr = [
        normalize_band(
            read_landsat_band_from_tif_file(
                image_path=image_path, 
                band_idx=band
            )
        ) 
        for band in bands
    ]

    return np.stack(_image_arr, axis=-1)

def read_tif_as_rgb_pillow_image(tif_path: str, bands: list):
    rgb_arr = process_single_image(tif_path, bands)
    pil_image_obj = PIL.Image.fromarray((rgb_arr * 255).astype(np.uint8))
    
    return pil_image_obj


def resample_real_esrgan(
            model, 
            filepath, 
            output_path,
            bands: list
    ) -> None:
        image = read_tif_as_rgb_pillow_image(tif_path=filepath, bands=bands)
        enhanced_image = model.predict(image)
        enhanced_image.save(output_path)

        print(f" -- saved image to {output_path}")


if __name__ == "__main__":
    ORIGINAL_IMAGES_FILE_PATH_ELE = ["/","home","farol-da-barra", "workspace", "hcordeiro", "aulas", "visao-computacional", "output-tiffs"]
    OUTPUT_PATH_FOR_PLOTS_ELE = ["/","home","farol-da-barra", "workspace", "hcordeiro", "aulas", "visao-computacional", "plots"]
    BANDS = [4, 3, 2]
    
    weights_path = "/home/farol-da-barra/workspace/hcordeiro/aulas/visao-computacional/weights/RealESRGAN_x4.pth"

    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(weights_path, download=True)

    print(" -- model and wheights loaded")

    base_path = os.path.join(*ORIGINAL_IMAGES_FILE_PATH_ELE)
    original_img_path_list = get_tif_images_path_list(base_path=base_path)

    for input_path in original_img_path_list:
        _out_file_name = os.path.basename(input_path)
        _out_file_name = os.path.splitext(_out_file_name)[0]
        print(f" -- processing {_out_file_name} ...")
        _out_file_name = f"enhanced_{_out_file_name}.png"

        _out_dir = "RealESRGAN"

        resample_real_esrgan(model=model, filepath=input_path, output_path=os.path.join(base_path, _out_dir, _out_file_name), bands=BANDS)