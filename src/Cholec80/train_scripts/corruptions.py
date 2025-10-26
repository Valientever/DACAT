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
import matplotlib
import uuid
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
    def to_numpy(img_tensor, apply_denorm=True):
        if img_tensor.dim() == 5:
            img_tensor = img_tensor[0, 0]  # (C, H, W)
        elif img_tensor.dim() == 4:
            img_tensor = img_tensor[0]     # (C, H, W)

        img_tensor = img_tensor.detach().cpu()

        # Denormalize if needed (assume ImageNet mean/std)
        if apply_denorm and torch.is_floating_point(img_tensor) and img_tensor.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        return img_np

    # --- Utility to save image ---
    def save_image(img_tensor, save_path, title="Image", apply_denorm=True):
        img_np = to_numpy(img_tensor, apply_denorm=apply_denorm)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # # Save all images
    base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/debug_folder/gaussian_noise"
    save_image(image, os.path.join(base_path, "original_image.png"), title="Original Image", apply_denorm=True)
    save_image(noise, os.path.join(base_path, "noise.png"), title="Noise", apply_denorm=False)
    save_image(noisy_image, os.path.join(base_path, "noisy_image.png"), title="Noisy Image", apply_denorm=True)

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
    # strength = 0.7
    # blurred_tensor = blurred_tensor * strength + original_tensor * (1 - strength)
    

    # ------------------ Visualization Block (non-invasive) ------------------

    # def to_numpy(img_tensor, denorm = True):
    #     img = img_tensor.detach().cpu()
    #     if img.dim() == 5:
    #         img = img[0, 0]
    #     elif img.dim() == 4:
    #         img = img[0]  # Take first image in batch

    #     print("Tensor stats:")
    #     print("  Shape:", img.shape)
    #     print("  Min:", img.min().item())
    #     print("  Max:", img.max().item())
    #     print("  Dtype:", img.dtype)

        
    #     # ✅ Apply ImageNet denormalization
    #     if denorm and img.shape[0] == 3:
    #         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #         img = img * std + mean


    #     img = img.permute(1, 2, 0).numpy()
    #     img = (img * 255).clip(0, 255).astype(np.uint8)
    #     return img
        

    # def save_image(img_np, path, title="Image"):
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     plt.imshow(img_np)
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.savefig(path, bbox_inches='tight')
    #     plt.close()

    # # Paths
    # base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/debug_folder/motion_blur"
    # save_image(to_numpy(original_tensor, denorm=True), os.path.join(base_path, "original_image_motion_blur.png"), title="Original Image")
    # save_image(to_numpy(blurred_tensor, denorm = True), os.path.join(base_path, "motion_blurred_image.png"), title="Motion Blurred Image")

    # # -----------------------------------------------------------------------

    return blurred_tensor


#Defocus blur
def apply_defocus_blur(image, kernel_size=15):
    is_batched = len(image.shape) == 5
    if is_batched:
        batch_size, timesteps, channels, height, width = image.shape
        image = image.view(-1, channels, height, width)  # Flatten batch & time

    original_tensor = image.clone()

    # Convert to (H, W, C) format for OpenCV
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B*T, H, W, C)

    # Apply Gaussian blur to each frame
    blurred_np = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in image_np])

    # Convert back to PyTorch tensor
    blurred_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(image.device).float()  # Shape: (B*T, C, H, W)

    # Reshape back if input was 5D
    if is_batched:
        blurred_tensor = blurred_tensor.view(batch_size, timesteps, channels, height, width)

    #Addign blur effect with a particular strength
    # strength = 0.5
    # blurred_tensor = blurred_tensor * strength + original_tensor * (1 - strength)
    

    # ------------------ Visualization Block (non-invasive) ------------------

    def to_numpy(img_tensor, denorm = True):
        img = img_tensor.detach().cpu()
        if img.dim() == 5:
            img = img[0, 0]
        elif img.dim() == 4:
            img = img[0]  # Take first image in batch

        print("Tensor stats:")
        print("  Shape:", img.shape)
        print("  Min:", img.min().item())
        print("  Max:", img.max().item())
        print("  Dtype:", img.dtype)

        
        # ✅ Apply ImageNet denormalization
        if denorm and img.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean


        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return img
        

    def save_image(img_np, path, title="Image"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # Paths
    base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/debug_folder/defocus_blur"
    save_image(to_numpy(original_tensor, denorm=True), os.path.join(base_path, "original_image_defocus_blur.png"), title="Original Image")
    save_image(to_numpy(blurred_tensor, denorm = True), os.path.join(base_path, "defocus_blurred_image.png"), title="Defocus Blurred Image")

    # -----------------------------------------------------------------------

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

    original_tensor = image.clone()

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

    # ------------------ Visualization Block (non-invasive) ------------------

    def to_numpy(img_tensor, denorm = True):
        img = img_tensor.detach().cpu()
        if img.dim() == 5:
            img = img[0, 0]
        elif img.dim() == 4:
            img = img[0]  # Take first image in batch

        # print("Tensor stats:")
        # print("  Shape:", img.shape)
        # print("  Min:", img.min().item())
        # print("  Max:", img.max().item())
        # print("  Dtype:", img.dtype)

        
    #     # ✅ Apply ImageNet denormalization
        if denorm and img.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean


        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return img
        

    def save_image(img_np, path, title="Image"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # Paths
    base_path = "/home/santhi/Documents/DACAT/src/Cholec80/results/debug_folder/uneven_illumination"
    save_image(to_numpy(original_tensor, denorm=True), os.path.join(base_path, "original_image.png"), title="Original Image")
    save_image(to_numpy(result_tensor, denorm = True), os.path.join(base_path, "ul_image.png"), title="Uneven Illumination Image")

    # -----------------------------------------------------------------------

    return result_tensor

#Smoke effect
# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate Perlin noise for corruption
def generate_perlin_noise(height, width, scale=10, intensity=0.5):
    if intensity is None:
        raise ValueError("Error: 'intensity' cannot be None. Please provide a valid float value.")
    
    # Try to import noise module again (for worker processes)
    try:
        import noise as noise_module
        noise_available = True
    except ImportError:
        noise_available = False
    
    if not noise_available:
        # Fallback to simple random noise if noise module is not available
        print("Warning: Using random noise instead of Perlin noise (noise module not available)")
        random_noise = np.random.rand(height, width).astype(np.float32)
        random_noise = random_noise * intensity
        return random_noise

    perlin_noise = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            perlin_noise[i, j] = noise_module.pnoise2(i / scale, j / scale, octaves=6)

    # Normalize to [0,1] and apply intensity
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
    perlin_noise = perlin_noise * intensity
    return perlin_noise

# Function to add realistic corruption (smoke effect)
def add_smoke_effect(image, intensity=0.7):
    """
    Apply smoke effect corruption to an image tensor.
    Accepts:
      - 3D:  (C,H,W) or (H,W,C)
      - 4D:  (N,C,H,W) or (N,H,W,C)
      - 5D:  (B,T,C,H,W) or (B,T,H,W,C)
    Returns a tensor with the SAME rank and channel/layout as the input.
    """
    # Input validation - same as other corruption functions
    if any(dim == 0 for dim in image.shape):
        print(f"add_smoke_effect: Received empty image with shape {image.shape} and dtype {image.dtype}. Stopping and returning None.")
        return None
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor")
    if intensity is None:
        raise ValueError("Error: 'intensity' must be a valid float value.")
    # print(f"Adding smoke effect with intensity {intensity} to image of shape {image.shape}")
    
    # Save original device
    original_device = image.device
    
    image = image.to('cpu').contiguous()
    orig_shape = image.shape
    orig_dtype = image.dtype
    squeeze_batch = False
    is_sequence = False
    if image.dim() == 5:
        b, t, c, h, w = image.shape
        image = image.view(-1, c, h, w)
        is_sequence = True
    elif image.dim() == 4:
        b, c, h, w = image.shape
    elif image.dim() == 3:
        c, h, w = image.shape
        image = image.unsqueeze(0)
        b = 1
        squeeze_batch = True
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if orig_dtype == torch.uint8:
        image_f32 = image.float() / 255.0
    else:
        image_f32 = image.float().clamp(0, 1)
    image_np = image_f32.permute(0, 2, 3, 1).cpu().numpy()
    b, h, w, c = image_np.shape
    if min(h, w, c) <= 0:
        print(f"Smoke effect received invalid image shape: {image_np.shape}. Stopping and returning None.")
        return None
    # Create a more diffuse, uniform smoke effect
    # Use higher scale for Perlin noise and higher sigma for Gaussian filter
    diffuse_scale = 500 #100 #max 500-1000 # Higher scale for larger, smoother noise features
    diffuse_sigma = 15  #15 #max 50-100 # Higher sigma for more diffusion
    # Optionally, add a random offset for each image in the batch
    noise_3ch = np.zeros((b, h, w, c), dtype=np.float32)
    for i in range(b):
        offset_x = np.random.randint(0, diffuse_scale)
        offset_y = np.random.randint(0, diffuse_scale)
        noise_pattern = generate_perlin_noise(h, w, scale=diffuse_scale, intensity=intensity)
        # Shift the noise pattern for each image for more variety
        noise_pattern = np.roll(noise_pattern, shift=offset_x, axis=0)
        noise_pattern = np.roll(noise_pattern, shift=offset_y, axis=1)
        noise_img = np.stack([noise_pattern] * c, axis=2)
        noise_img = gaussian_filter(noise_img, sigma=diffuse_sigma)
        noise_3ch[i] = noise_img
    corrupted = cv2.addWeighted(image_np.astype(np.float32), 1.0 - intensity, noise_3ch.astype(np.float32), intensity, 0)
    corrupted = np.clip(corrupted, 0, 1)
    corrupted_tensor = torch.from_numpy(corrupted).permute(0, 3, 1, 2).contiguous()
    if orig_dtype == torch.uint8:
        corrupted_tensor = (corrupted_tensor * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        corrupted_tensor = corrupted_tensor.float().clamp(0, 1)
    if is_sequence:
        corrupted_tensor = corrupted_tensor.view(orig_shape)
    elif squeeze_batch:
        corrupted_tensor = corrupted_tensor.squeeze(0)
    out_shape = tuple(corrupted_tensor.shape)
    if any(dim == 0 for dim in out_shape):
        # print(f"Smoke effect produced empty image with shape {out_shape} and dtype {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    if torch.isnan(corrupted_tensor).any():
        print(f"Smoke effect produced image with NaNs. Shape: {out_shape}, dtype: {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    min_val = corrupted_tensor.min().item() if corrupted_tensor.numel() > 0 else None
    max_val = corrupted_tensor.max().item() if corrupted_tensor.numel() > 0 else None
    # print(f"Smoke effect output shape: {out_shape}, dtype: {corrupted_tensor.dtype}, min: {min_val}, max: {max_val}")
    if min_val == 0 and max_val == 0:
        print(f"Smoke effect produced all-zero image. Shape: {out_shape}, dtype: {corrupted_tensor.dtype}. Stopping and returning None.")
        return None
    # Now do plotting after all error checks
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    global _smoke_effect_save_counter
    if '_smoke_effect_save_counter' not in globals():
        _smoke_effect_save_counter = 0
    if _smoke_effect_save_counter < 1:
        # Get the original input for visualization (before flattening)
        inp_np = image_f32.detach().cpu().numpy()
        out_np = corrupted_tensor.detach().cpu().numpy()
        
        # Handle different tensor shapes - always extract first sample
        # If 5D (from is_sequence), select first batch and first timestep
        if out_np.ndim == 5:
            out_np = out_np[0, 0]  # (B, T, C, H, W) -> (C, H, W)
        # If 4D, select first batch
        elif out_np.ndim == 4:
            out_np = out_np[0]  # (B, C, H, W) -> (C, H, W)
        
        # Do the same for input
        if inp_np.ndim == 4:
            inp_np = inp_np[0]  # (B, C, H, W) -> (C, H, W)
        
        # If channel-first, transpose to HWC
        if inp_np.ndim == 3 and inp_np.shape[0] in [1,3]:
            inp_np = inp_np.transpose(1,2,0)
        if out_np.ndim == 3 and out_np.shape[0] in [1,3]:
            out_np = out_np.transpose(1,2,0)
        
        # Convert to uint8 for display
        if inp_np.dtype != np.uint8:
            inp_np = (inp_np * 255).clip(0, 255).astype(np.uint8)
        if out_np.dtype != np.uint8:
            out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
        
        debug_dir = '/home/santhi/Documents/DACAT/src/Cholec80/results/debug_folder/smoke_effect'
        os.makedirs(debug_dir, exist_ok=True)
        unique_id = str(uuid.uuid4())
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title('Input Image')
        plt.imshow(inp_np)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Smoke Effect Output')
        plt.imshow(out_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{debug_dir}/input_and_se_{intensity}_{diffuse_scale}_{diffuse_sigma}_{unique_id}.png')
        plt.close()
        _smoke_effect_save_counter += 1
    # --------------------------------------------------------
    
    # Move tensor back to original device before returning
    corrupted_tensor = corrupted_tensor.to(original_device)
    
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
    elif corruption == 'clean' or corruption == "none" or corruption is None:
        return image  # No corruption applied
    else:
        # print("no corruption being called")
        # return image
        raise ValueError(f"Invalid corruption type '{corruption}'. Choose from: 'gaussian_noise', 'motion_blur', 'defocus_blur', 'uneven_illumination', 'smoke_effect', 'random'.")