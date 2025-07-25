import numpy as np
import random
import time
from scipy.ndimage import zoom

def augment(feature_input: list):
    res = []
    res.append(feature_input)
    #res.append(vertical_flip(feature_input))
    #res.append(horizontal_flip(feature_input))
    res.extend(noise(feature_input))

    return res

def vertical_flip(feature_input: list):
    return feature_input

def horizontal_flip(feature_input: list):
    return feature_input

def noise(feature_input: list):
    n = len(feature_input)
    num_iterations = 8 # augment data by factor of 128
    bucket_size = 256 # add noise to 1 out of every 16 elements in features
    res = []

    for i in range(num_iterations):
        copy = feature_input.copy()
        for j in range(0, n // bucket_size):
            rand_index = random.randint(j * bucket_size, (j + 1) * bucket_size - 1)
            copy[rand_index] = copy[rand_index] + 1
        
        res.append(copy)

    return res

def horizontal_flip(feature_input: np.ndarray):
    if feature_input.shape != (3, 32, 32):
        raise ValueError(f"Expected shape (3, 32, 32), got {feature_input.shape}")
    
    return feature_input[:, :, ::-1]  # Flip along the width dimension

def vertical_flip(feature_input: np.ndarray):
    if feature_input.shape != (3, 32, 32):
        raise ValueError(f"Expected shape (3, 32, 32), got {feature_input.shape}")
    
    return feature_input[:, ::-1, :]

def scale(feature_input: np.ndarray, scale: float = 1.1):
    if feature_input.shape != (3, 32, 32):
        raise ValueError(f"Expected shape (3, 32, 32), got {feature_input.shape}")
    
    scaled = zoom(feature_input, (1.0, 1.1, 1.1), order=1)  # order=1 for bilinear
    
    h, w = scaled.shape[1], scaled.shape[2]
    
    start_h = (h - 32) // 2
    start_w = (w - 32) // 2
    end_h = start_h + 32
    end_w = start_w + 32
    
    cropped = scaled[:, start_h:end_h, start_w:end_w]
    return cropped


def noise_conv_2d(feature_input: np.ndarray):
    
    # Ensure the input has the correct shape (3, 32, 32)
    if feature_input.shape != (3, 32, 32):
        raise ValueError(f"Expected shape (3, 32, 32), got {feature_input.shape}")
    
    num_iterations = 4  # augment data by factor of 8
    res = []
    
    for i in range(num_iterations):
        # Create a copy of the input
        copy = feature_input.copy()
        num_noise_pixels = 16  # Add noise to 16 random pixels
        for _ in range(num_noise_pixels):
            # Random channel (0, 1, or 2 for RGB)
            channel = random.randint(0, 2)
            # Random spatial coordinates
            row = random.randint(0, 31)
            col = random.randint(0, 31)
            
            # Add small random noise (between -0.1 and 0.1)
            noise_value = random.uniform(-0.025, 0.025)
            copy[channel, row, col] += noise_value
            
            # Clamp values to valid range [0, 1]
            copy[channel, row, col] = np.clip(copy[channel, row, col], 0.0, 1.0)
        
        res.append(copy)
    
    return res
    






