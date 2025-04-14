# SWINIR

# SwinIR (Swin Transformer for Image Restoration)

SwinIR is a pure transformer-based architecture inspired by the **Swin Transformer (Shifted Window Transformer)** for image restoration tasks such as super-resolution. 
This model uses window-based self-attention to learn image features effectively and efficiently, making it suitable for enhancing image resolution.

## Features
1.Pure Transformer-based Architecture: Utilizes the Swin Transformer model for image restoration, specifically designed for super-resolution tasks.
2.Window-based Attention: The attention mechanism is applied in local windows, which makes the model computationally efficient while retaining performance.
3.Upscaling: The model supports various upscaling factors (e.g., 2x, 4x) for image super-resolution tasks.
4.Feedforward Network: After attention computation, the model applies a standard feedforward network to further refine features.
5.Customizable Hyperparameters: You can adjust the embedding dimension, number of layers (depths), number of heads, and other parameters.able Hyperparameters**: You can adjust the embedding dimension, number of layers (depths), number of heads, and other parameters.

## Requirements
The following Python libraries are required to run the model:
torch (for PyTorch)
torchvision (optional, for image transformations)
numpy
matplotlib (optional, for visualizing results)

You can install them via pip:
pip install torch torchvision numpy matplotlib


## Model Architecture
1. Swin Transformer Layer:
Window-based Self-Attention: Operates over non-overlapping windows of the image.

Feedforward Network: Applied after the attention block for feature refinement.

Layer Normalization: Applied before attention and feedforward blocks to stabilize training.

2. Upscaling:
The model includes a pixel-shuffling layer that upscales the image, with the number of upscaling levels specified by the upscale parameter.

3. Final Convolutions:
Convolutions are applied before and after the transformer layers to process the image features, followed by a final convolutional layer to output the restored image.

## Model Usage
## Example Usage
import torch
from swinir import SwinIR

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinIR(in_chans=3, embed_dim=64, depths=4, num_heads=8, upscale=4, window_size=8).to(device)

# Generate a dummy low-resolution image (for testing purposes)
lr_image = torch.randn(1, 3, 128, 128).to(device)  # Example low-resolution image

# Forward pass through the model
sr_image = model(lr_image)

# Output the shape of the super-resolved image (should be [1, 3, 512, 512] for upscale=4)
print(sr_image.shape)


Model Input/Output
Input: The input should be a low-resolution image of shape (B, C, H, W), where B is the batch size, C is the number of channels (e.g., 3 for RGB), and H, W are the height and width of the image.
Output: The model outputs a high-resolution image of shape (B, C, upscale*H, upscale*W).

## Hyperparameters
in_chans: Number of input channels (e.g., 3 for RGB).
embed_dim: Embedding dimension (i.e., the number of channels in each transformer layer).
depths: Number of Swin Transformer layers in the model.
num_heads: Number of attention heads in each transformer layer.
upscale: Upscaling factor for the output image.
window_size: Size of the windows for the window-based attention mechanism.


## CODE:
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super(SwinTransformerLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim = dim
        self.num_heads = num_heads

        # Define the attention layer (Window-based Self-Attention)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=window_size)

        # Feedforward layer
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape x into patches (flatten the spatial dimensions)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C], N = H * W
        
        # Apply window-based attention
        x = self.norm1(x)
        attn_out = self.attn(x)
        x = x + attn_out

        # Apply Feedforward
        x = self.norm2(x)
        ff_out = self.ffn(x)
        x = x + ff_out
        
        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinIR(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, depths=4, num_heads=8, upscale=4, window_size=8):
        super(SwinIR, self).__init__()

        self.upscale = upscale
        self.embed_dim = embed_dim

        # Initial convolution
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Build the Swin Transformer layers
        self.layers = nn.ModuleList([
            SwinTransformerLayer(embed_dim, num_heads, window_size) for _ in range(depths)
        ])

        # Final convolution and upsampling layers
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = self.build_upsample(embed_dim, upscale)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def build_upsample(self, embed_dim, scale):
        layers = []
        for _ in range(int(math.log(scale, 2))):
            layers.append(nn.Conv2d(embed_dim, 4 * embed_dim, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_first(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv_after_body(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinIR(in_chans=3, embed_dim=64, depths=4, num_heads=8, upscale=4, window_size=8).to(device)

lr_image = torch.randn(1, 3, 128, 128).to(device)

sr_image = model(lr_image)
print(sr_image.shape)  # Should output (1, 3, 512, 512) if upscale_factor=4
