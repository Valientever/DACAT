#corruptions
import cv2
import numpy as np
from ipdb import set_trace
import torch
import numpy as np
import cv2

#Gaussian noise
def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to a PyTorch image tensor without changing its shape or type.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy image tensor with the same shape and type as the input.
    """
    # Ensure noise has the same shape as the image
    noise = torch.randn_like(image) * std + mean  # Gaussian noise

    # Add noise and retain original type
    noisy_image = image + noise

    # Clip values to stay within valid range if input is uint8 (0-255)
    if image.dtype == torch.uint8:
        noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    return noisy_image



#Motion blur
def apply_motion_blur(image, kernel_size=15):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

    blurred = cv2.filter2D(image_np, -1, kernel)

    blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).to(image.device).float()
    return blurred_tensor

#Defocus blur
def apply_defocus_blur(image, kernel_size=15):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)

    blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).to(image.device).float()
    return blurred_tensor


#Ajust illumination
def uneven_illumination(image, strength=0.5):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    h, w, c = image_np.shape
    gradient = np.linspace(1 - strength, 1, w, dtype=np.float32)
    gradient = np.tile(gradient, (h, 1)).reshape(h, w, 1)

    illuminated = image_np * gradient
    illuminated = np.clip(illuminated, 0, 1)

    illuminated_tensor = torch.from_numpy(illuminated).permute(2, 0, 1).to(image.device).float()
    return illuminated_tensor

#Smoke effect
def add_smoke_effect(image, intensity=0.5):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    h, w, c = image_np.shape
    smoke = np.random.normal(loc=0.5, scale=intensity, size=(h, w, c)).astype(np.float32)

    smoked = cv2.addWeighted(image_np, 1 - intensity, smoke, intensity, 0)
    smoked = np.clip(smoked, 0, 1)

    smoked_tensor = torch.from_numpy(smoked).permute(2, 0, 1).to(image.device).float()
    return smoked_tensor


def corruption(image, corruption_type):
    if corruption_type == 'gaussian_noise':
        return add_gaussian_noise(image)
    elif corruption_type == 'motion_blur':
        return apply_motion_blur(image)
    elif corruption_type == 'defocus_blur':
        return apply_defocus_blur(image)
    elif corruption_type == 'uneven_illumination':
        return uneven_illumination(image)
    elif corruption_type == 'smoke_effect':
        smoke_overlay = cv2.imread('smoke.png', cv2.IMREAD_UNCHANGED)
        return add_smoke_effect(image, smoke_overlay)
    else:
        return image