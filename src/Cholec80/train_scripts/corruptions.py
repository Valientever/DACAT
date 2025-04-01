#corruptions
import cv2
import numpy as np
from ipdb import set_trace
import torch
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import noise
import os
from scipy.ndimage import gaussian_filter
import random
import matplotlib.pyplot as plt
#Gaussian noise
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=0.5):
    """
    Adds Gaussian noise to a PyTorch image tensor without changing its shape or type.
    Also saves the original image, noise, and noisy image to disk.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W), (B, C, H, W), or (B, T, C, H, W).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy image tensor with the same shape and type as the input.
    """
    # Generate Gaussian noise with same shape
    noise = torch.randn_like(image, dtype=torch.float32) * std + mean
    noisy_image = image.float() + noise

    # Clip and convert back to uint8 if needed
    if image.dtype == torch.uint8:
        noisy_image = torch.clamp(noisy_image, 0, 255).to(torch.uint8)

    # --- Utility to convert tensor to numpy for visualization ---
    # def to_numpy(img_tensor, apply_denorm=True):
    #     if img_tensor.dim() == 5:
    #         img_tensor = img_tensor[0, 0]  # (C, H, W)
    #     elif img_tensor.dim() == 4:
    #         img_tensor = img_tensor[0]     # (C, H, W)

    #     img_tensor = img_tensor.detach().cpu()

    # #     # Denormalize if needed (assume ImageNet mean/std)
    #     if apply_denorm and torch.is_floating_point(img_tensor) and img_tensor.shape[0] == 3:
    #         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #         img_tensor = img_tensor * std + mean

    #     img_np = img_tensor.permute(1, 2, 0).numpy()
    #     img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    #     return img_np

    # # --- Utility to save image ---
    # def save_image(img_tensor, save_path, title="Image", apply_denorm=True):
    #     img_np = to_numpy(img_tensor, apply_denorm=apply_denorm)
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.imshow(img_np)
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.savefig(save_path, bbox_inches='tight')
    #     plt.close()

    # # Save all images
    # base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/check"
    # save_image(image, os.path.join(base_path, "original_image.png"), title="Original Image", apply_denorm=True)
    # save_image(noise, os.path.join(base_path, "noise.png"), title="Noise", apply_denorm=False)
    # save_image(noisy_image, os.path.join(base_path, "noisy_image.png"), title="Noisy Image", apply_denorm=True)

    # # Print value ranges
    # print("Original range:", image.min().item(), "-", image.max().item())
    # print("Noise range:", noise.min().item(), "-", noise.max().item())
    # print("Noisy range:", noisy_image.min().item(), "-", noisy_image.max().item())

    return noisy_image





#Motion blur

import matplotlib.pyplot as plt
import os

def apply_motion_blur(image, kernel_size=15):
    # Check if the input is batched (5D: B, T, C, H, W)
    is_batched = len(image.shape) == 5
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Reshape to (B*T, C, H, W)

    # Save original for visualization
    original_tensor = image.clone()

    # Convert to (H, W, C) format for OpenCV processing
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # (B*T, H, W, C)

    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size

    # Apply motion blur to each image in the batch
    blurred_np = np.array([cv2.filter2D(img, -1, kernel) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()  # (B*T, C, H, W)

    # Reshape back if input was 5D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)

    #Addign blur effect with a particular strength
    strength = 0.7
    blurred_tensor = blurred_tensor * strength + original_tensor * (1 - strength)
    

    # ------------------ Visualization Block (non-invasive) ------------------

    # def to_numpy(img_tensor, denorm = True):
    #     img = img_tensor.detach().cpu()
    #     if img.dim() == 5:
    #         img = img[0, 0]
    #     elif img.dim() == 4:
    #         img = img[0]  # Take first image in batch

        # print("Tensor stats:")
        # print("  Shape:", img.shape)
        # print("  Min:", img.min().item())
        # print("  Max:", img.max().item())
        # print("  Dtype:", img.dtype)

        
        # ✅ Apply ImageNet denormalization
        # if denorm and img.shape[0] == 3:
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        #     img = img * std + mean


        # img = img.permute(1, 2, 0).numpy()
        # img = (img * 255).clip(0, 255).astype(np.uint8)
        # return img
        

    # def save_image(img_np, path, title="Image"):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     plt.imshow(img_np)
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()

    # Paths
    # base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/check_mb"
    # save_image(to_numpy(original_tensor, denorm=True), os.path.join(base_path, "original_image_motion_blur.png"), title="Original Image")
    # save_image(to_numpy(blurred_tensor, denorm = True), os.path.join(base_path, "motion_blurred_image.png"), title="Motion Blurred Image")

    # -----------------------------------------------------------------------

    return blurred_tensor


#Defocus blur
def apply_defocus_blur(image, kernel_size=15):
    is_batched = len(image.shape) == 5
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Flatten batch & time

    # Convert to (H, W, C) format for OpenCV
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B*T, H, W, C)

    # Apply Gaussian blur to each frame
    blurred_np = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()  # Shape: (B*T, C, H, W)

    # Reshape back if input was 5D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)

    return blurred_tensor

#Ajust illumination
def uneven_illumination(image, strength=0.5):
    # Check if input is 5D (batch + time)
    is_batched = (image.dim() == 5)
    if is_batched:
        b, t, c, h, w = image.shape
        # Flatten (B, T) into one dimension => (B*T, C, H, W)
        image = image.view(-1, c, h, w)
    else:
        # Single image: (C, H, W)
        c, h, w = image.shape
        b, t = 1, 1  # For convenient unflattening logic at the end

    # Convert to (N, H, W, C) for OpenCV-like operations
    # N = B*T for batched input, or 1 for single input
    image_np = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    N = image_np.shape[0]  # number of images/frames

    # Apply uneven illumination to each frame
    result_list = []
    for i in range(N):
        # Get one frame: shape (H, W, C)
        frame = image_np[i]
        h_i, w_i, c_i = frame.shape

        # Create horizontal gradient from (1 - strength) to 1
        gradient = np.linspace(1 - strength, 1, w_i, dtype=np.float32)
        # Tile vertically to match frame height, shape => (H, W, 1)
        gradient = np.tile(gradient, (h_i, 1)).reshape(h_i, w_i, 1)

        # Multiply frame by gradient, then clip to [0, 1]
        illuminated = frame * gradient
        illuminated = np.clip(illuminated, 0, 1)
        
        result_list.append(illuminated)

    # Stack all processed frames back into shape (N, H, W, C)
    result_np = np.stack(result_list, axis=0)

    # Convert to torch => shape (N, C, H, W)
    result_tensor = torch.from_numpy(result_np).permute(0, 3, 1, 2).to(image.device).float()

    # Unflatten if originally batched
    if is_batched:
        result_tensor = result_tensor.view(b, t, c, h, w)

    return result_tensor

#Smoke effect
# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate Perlin noise for corruption
def generate_perlin_noise(height, width, scale=10, intensity=0.5):
    if intensity is None:
        raise ValueError("Error: 'intensity' cannot be None. Please provide a valid float value.")

    perlin_noise = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            perlin_noise[i, j] = noise.pnoise2(i / scale, j / scale, octaves=6)

    # Normalize to [0,1] and apply intensity
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
    perlin_noise = perlin_noise * intensity
    return perlin_noise

# Function to add realistic corruption (smoke effect)
def add_smoke_effect(image, intensity=0.7):
    """
    Apply a realistic smoke effect to an image tensor while handling different tensor shapes.

    Expected Input Shapes:
    - [C, H, W] → Single Image
    - [B, C, H, W] → Batch of Images
    - [B, S, C, H, W] → Batch with Sequence Length (Fix Applied)
    """

    # Ensure input is a tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor")

    # Ensure intensity is a valid float
    if intensity is None:
        raise ValueError("Error: 'intensity' must be a valid float value.")

    # Move image to the correct device
    image = image.to(device).clone()

    # Handle 5D tensor (Batch, Sequence, Channels, Height, Width)
    is_sequence = False
    if image.dim() == 5:  
        # print(f"Received a 5D tensor with shape: {image.shape}")
        batch_size, seq_len, channels, height, width = image.shape
        image = image.view(batch_size * seq_len, channels, height, width)  # Flatten the sequence dimension
        is_sequence = True  # Track that this was a sequence

    # Handle 4D tensor (Batch, Channels, Height, Width)
    if image.dim() == 4:
        # print(f"Processing batch of images with shape: {image.shape}")
        batch_size, channels, height, width = image.shape
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (B, H, W, C)
    else:
        # Handle single image (3D tensor: [C, H, W])
        # print(f"Processing single image with shape: {image.shape}")
        image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)

    h, w = image_np.shape[-3:-1]

    # Generate Perlin noise for smoke
    noise_pattern = generate_perlin_noise(h, w, scale=50, intensity=intensity)

    # Convert to 3 channels and apply Gaussian blur for a natural effect
    noise_3ch = np.stack([noise_pattern] * 3, axis=-1)
    noise_3ch = gaussian_filter(noise_3ch, sigma=5)

    # **Fix: Ensure noise shape matches image_np (including batch dimension)**
    if image_np.ndim == 4:  # If batch is present
        # print(f"Reshaping noise to match batch shape: {image_np.shape}")
        noise_3ch = np.expand_dims(noise_3ch, axis=0)  # Add batch dimension
        noise_3ch = np.repeat(noise_3ch, batch_size, axis=0)  # Match batch size

    # Ensure the shape of noise_3ch matches image_np
    assert noise_3ch.shape == image_np.shape, f"Shape mismatch: noise {noise_3ch.shape} vs image {image_np.shape}"

    # Blend noise with the image while keeping original dimensions
    corrupted = cv2.addWeighted(image_np, 1.0 - intensity, noise_3ch, intensity, 0)
    corrupted = np.clip(corrupted, 0, 1)

    # Convert back to tensor and reshape if necessary
    corrupted_tensor = torch.from_numpy(corrupted).permute(0, 3, 1, 2).to(device).float()  # Back to (B, C, H, W)

    # **Fix: Ensure final output is in [B, 3, H, W]**
    if corrupted_tensor.shape[1] != 3:
        # print(f"Fixing incorrect channel count: {corrupted_tensor.shape[1]} → 3")
        corrupted_tensor = corrupted_tensor[:, :3, :, :]  # Ensure only 3 channels

    # If input was 5D, restore sequence shape
    if is_sequence:
        corrupted_tensor = corrupted_tensor.view(batch_size // seq_len, seq_len, 3, height, width)

    return corrupted_tensor

import random

# Global counter for verification
corruption_tracker = {"total": 0, "corrupted": 0, "uncorrupted": 0}

def random_corrupt(image):
    """
    Applies a single random corruption to an image with a 50% probability.
    Keeps track of how many images are corrupted.
    """

    corruption_methods = [
        add_gaussian_noise,
        apply_motion_blur,
        apply_defocus_blur,
        uneven_illumination,
        add_smoke_effect
    ]

    corruption_tracker["total"] += 1  # Track total processed images
    # set_trace()
    if random.random() < 0.5:  # 50% chance to apply corruption
        corruption_method = random.choice(corruption_methods)
        image = corruption_method(image)
        corruption_tracker["corrupted"] += 1  # Track corrupted images
        # print(f"✅ Image CORRUPTED using {corruption_method.__name__}")
    else:
        corruption_tracker["uncorrupted"] += 1  # Track uncorrupted images
        # print("❌ Image left UNCHANGED")
    # print(f"Total: {corruption_tracker['total']} | Corrupted: {corruption_tracker['corrupted']} | Uncorrupted: {corruption_tracker['uncorrupted']}")
    return image  # Return the (possibly corrupted) image


def corruption(image, corruption):
    if corruption == 'gaussian_noise':
        return add_gaussian_noise(image)
    elif corruption == 'motion_blur':
        return apply_motion_blur(image)
    elif corruption == 'defocus_blur':
        return apply_defocus_blur(image)
    elif corruption == 'uneven_illumination':
        return uneven_illumination(image)
    elif corruption == 'smoke_effect':
        return add_smoke_effect(image, intensity=0.7)
    elif corruption == 'random':
        return random_corrupt(image)
    else:
        # print("no corruption being called")
        # return image
        raise ValueError(f"Invalid corruption type '{corruption}'. Choose from: 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random'.")