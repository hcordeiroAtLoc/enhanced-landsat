import os
import rasterio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image


ORIGINAL_IMAGES_FILE_PATH_ELE = ["/","home","farol-da-barra", "workspace", "hcordeiro", "aulas", "visao-computacional", "output-tiffs"]
ENHANCED_IMAGE_DIR_PATH_ELE = ["10m", "OpenCV"]
LANDSAT_BAND_IDX_DICT = {
    "red": 4,
    "green": 3,
    "blue": 2,
}

OUTPUT_PATH_FOR_PLOTS_ELE = ["/","home","farol-da-barra", "workspace", "hcordeiro", "aulas", "visao-computacional", "plots"]
PLOT_TITLE = "Landsat Original Image (30m) and Enhanced Resolution (10m): Lanczos Method"


def read_landsat_band_from_tif_file(image_path: str, band_idx: int):
    with rasterio.open(image_path) as src:
        return src.read(band_idx)


def normalize_band(band):
    assert band.any(), """band needs non-na value. band.any() returned False"""
    # min-max scaling
    return (band - band.min()) / (band.max() - band.min())


def get_tif_images_path_list(base_path: str) -> list:
    tif_paths = [f.path 
                 for f in os.scandir(base_path) 
                 if f.is_file() and os.path.splitext(f.name)[1] == ".tif" 
                 ]
    
    return tif_paths


def get_enhanced_images_path_list(base_path: str, *paths) -> list:
    _numbered_image_dirs = [f.path 
                   for f in os.scandir(base_path) 
                   if f.is_dir()
                   ]  # img dirs are numbered, like 1_01 (area-idx_img-idx)
    
    enhanced_images_full_paths = []
    for dir in _numbered_image_dirs:
        target_path = os.path.join(dir, *paths)
        
        if os.path.exists(target_path) and os.path.isdir(target_path):
            img_paths = get_tif_images_path_list(target_path)
            enhanced_images_full_paths.extend(img_paths)
    
    return enhanced_images_full_paths


def create_split_view(original_image: np.ndarray, enhanced_image: np.ndarray):
    """
    Creates a split view of two images by combining the left half of the first image
    and the right half of the second image.
    
    Parameters:
    image1 (np.ndarray): First image as a numpy array.
    image2 (np.ndarray): Second image as a numpy array.
    
    Returns:
    np.ndarray: The combined split view image.
    """
    # Convert images to PIL for resizing
    img1_pil = Image.fromarray((original_image * 255).astype(np.uint8))
    img2_pil = Image.fromarray((enhanced_image * 255).astype(np.uint8))

    # Resize images to have the same height
    common_height = max(img1_pil.height, img2_pil.height)
    common_width = max(img1_pil.width, img2_pil.width)
    img1_resized = img1_pil.resize((common_width, common_height), resample=Image.NEAREST)
    img2_resized = img2_pil.resize((common_width, common_height), resample=Image.NEAREST)

    # Convert back to numpy arrays
    img1_resized = np.array(img1_resized)
    img2_resized = np.array(img2_resized)

    # Take left half of the first image and right half of the second image
    if common_width > common_height:
        half_width1 = img1_resized.shape[1] // 2
        half_width2 = img2_resized.shape[1] // 2
        split_view = np.hstack((
            img1_resized[:, :half_width1], 
            img2_resized[:, -half_width2:]
            ))

        return split_view, common_width, common_height
    
    half_height1 = img1_resized.shape[0] // 2
    half_height2 = img2_resized.shape[0] // 2
    split_view = np.vstack((
        img1_resized[:half_height1, :], 
        img2_resized[-half_height2:, :]
        ))

    return split_view, common_width, common_height


def process_single_image(image_path: str, bands: list):
    # que lindinho
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


def plot_image(
    image: np.ndarray,
    title: str, 
    img_width: int, 
    img_height: int, 
    output_path: str = None, 
    is_save: bool = False
) -> None:
    """
    Generates a plot of an image with a line added (vertical or horizontal based on image dimensions).
    Optionally saves the plot as a PNG instead of displaying it.
    
    Parameters:
    - image (np.ndarray): The image data to plot.
    - img_width (int): The width of the image.
    - img_height (int): The height of the image.
    - output_path (str, optional): File path to save the image (required if is_save=True).
    - is_save (bool): Whether to save the plot as a PNG (default is False).
    
    Returns:
    - None
    """
    # Create the plot
    fig = px.imshow(image)
    
    # Add a vertical or horizontal line
    if img_width > img_height:
        # Vertical line
        fig.add_shape(
            type="line",
            x0=img_width // 2, y0=0,
            x1=img_width // 2, y1=img_height,
            line=dict(color="red", width=5)
        )
    else:
        # Horizontal line
        fig.add_shape(
            type="line",
            x0=0, y0=img_height // 2,
            x1=img_width, y1=img_height // 2,
            line=dict(color="red", width=5)
        )
    
    # Update layout for a prettier plot
    fig.update_layout(
        coloraxis_showscale=False,  # Remove color scale
        title=dict(
            text=title,
            font=dict(size=20), 
            x=0.5  # Center the title
        ),
        margin=dict(l=10, r=10, t=50, b=10)  # Adjust margins
    )
    
    # Save or display the plot
    if is_save:
        if not output_path:
            raise ValueError("An output path must be provided when is_save is True.")
        fig.write_image(output_path, format='png', width=800, height=600)
        print(f"Plot saved to {output_path}")
    else:
        fig.show()

    
def process_collection(output_base_path: str) -> None:
    base_path = os.path.join(*ORIGINAL_IMAGES_FILE_PATH_ELE)
    bands = [4, 3, 2]

    original_img_path_list = get_tif_images_path_list(base_path=base_path)
    original_img_filename_list = [os.path.basename(fname) for fname in original_img_path_list]
    original_imgs_path_dict = {k:v for k, v in zip(original_img_filename_list, original_img_path_list)}
    
    enhanced_img_path_list = get_enhanced_images_path_list(base_path, *ENHANCED_IMAGE_DIR_PATH_ELE)
    enhanced_img_filename_list = [os.path.basename(fname) for fname in enhanced_img_path_list]
    enhanced_imgs_path_dict = {k:v for k, v in zip(enhanced_img_filename_list, enhanced_img_path_list)}

    # pair og paths with enhanced paths
    paired_img_tuple_list = []
    for fname in original_img_filename_list:
        path_tuple = (original_imgs_path_dict[fname], enhanced_imgs_path_dict[fname])
        paired_img_tuple_list.append(path_tuple)

    for original_path, enhanced_path in paired_img_tuple_list:
        original_img = process_single_image(original_path, bands)
        enhanced_img = process_single_image(enhanced_path, bands)

        splitted_img, img_width, img_height = create_split_view(original_img, enhanced_img)
        
        _out_file_name = os.path.basename(original_path)
        _out_file_name = os.path.splitext(_out_file_name)[0]
        _out_file_name = f"{_out_file_name}.png"

        plot_image(
            image=splitted_img,
            title=PLOT_TITLE,
            img_width=img_width,
            img_height=img_height,
            is_save=False,
            output_path=os.path.join(output_base_path, _out_file_name)
        )
    
    return None


if __name__ == "__main__":
    output_base_path = os.path.join(*OUTPUT_PATH_FOR_PLOTS_ELE)
    process_collection(output_base_path)