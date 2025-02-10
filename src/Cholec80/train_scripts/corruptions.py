#corruptions
import cv2
import numpy as np



#Noise
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

#Motion blur
def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

#Defocus blur
def apply_defocus_blur(image, kernel_size=15):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


#Ajust illumination
def adjust_brightness_contrast(image, brightness=30, contrast=50):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast / 127 + 1, beta=brightness)
    return adjusted

#Smoke effect
def add_smoke_effect(image, smoke_overlay):
    alpha = 0.5  # Adjust transparency
    smoke_overlay = cv2.GaussianBlur(smoke_overlay, (15, 15), 0)
    if smoke_overlay.shape[:2] != image.shape[:2]:
        smoke_overlay = cv2.resize(smoke_overlay, (image.shape[1], image.shape[0]))
    combined = cv2.addWeighted(image, 1-alpha, smoke_overlay, alpha, 0)
    return combined


def corruption(image, corruption_type):
    if corruption_type == 'gaussian_noise':
        return add_gaussian_noise(image)
    elif corruption_type == 'motion_blur':
        return apply_motion_blur(image)
    elif corruption_type == 'defocus_blur':
        return apply_defocus_blur(image)
    elif corruption_type == 'brightness_contrast':
        return adjust_brightness_contrast(image)
    elif corruption_type == 'smoke_effect':
        smoke_overlay = cv2.imread('smoke.png', cv2.IMREAD_UNCHANGED)
        return add_smoke_effect(image, smoke_overlay)
    else:
        return image