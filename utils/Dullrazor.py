
import matplotlib.pyplot as plt 
import torch    
import torch.nn.functional as F

def blackhat_transform(gray_tensor, kernel_size=9):
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=gray_tensor.device)
    dilated = F.max_pool2d(gray_tensor, kernel_size, 1, padding)
    closed = -F.max_pool2d(-dilated, kernel_size, 1, padding)
    blackhat = (closed - gray_tensor).clamp(0, 1)
    return blackhat

def patch_fill(image, mask, kernel_size=15):
    padding = kernel_size // 2
    ones = torch.ones_like(image)
    masked_input = image * (1 - mask)
    norm = F.avg_pool2d(ones * (1 - mask), kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm
    return masked_input + smooth * mask

def dullrazor(img, th=0.05):

    H, W, C = img.shape
    img = img.unsqueeze(0)  # -> [1, C, H, W]

    # Convert to grayscale using standard RGB weighting
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
    gray = gray.unsqueeze(1)  # -> [1, 1, H, W]

    blackhat = blackhat_transform(gray, kernel_size=9)
    hair_mask = (blackhat > th).float()
    hair_mask_3ch = hair_mask.repeat(1, 3, 1, 1)

    clean = patch_fill(img, hair_mask_3ch)  # -> [1, 3, H, W]
    return clean.squeeze(0).clamp(0, 1) 
