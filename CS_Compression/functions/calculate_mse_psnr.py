from PIL import Image
import numpy as np

def calculate_mse_psnr(image1, image2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
    image1_path (str): Path to the first image.
    image2_path (str): Path to the second image.

    Returns:
    float: The PSNR value.
    """
    # Open images and convert them to 'RGB' mode
    img1_np = np.array(image1).astype(float)
    img2_np = np.array(image2).astype(float)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1_np - img2_np) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return mse,psnr