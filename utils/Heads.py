import torch.nn.functional as F
import math

from torch import nn 

def get_teacher_momentum(current_epoch, max_epochs, base_m=0.996, final_m=1.0):
    """Cosine schedule used by DINO: base momentum approaches one smoothly."""
    if max_epochs <= 1:
        return final_m
    progress = current_epoch / (max_epochs - 1)
    return final_m - (final_m - base_m) * (math.cos(math.pi * progress) + 1.0) / 2.0

def get_teacher_temp(epoch, warmup_epochs=20, final_temp=0.07):
    start_temp = 0.04
    if epoch < warmup_epochs:
        return start_temp + (final_temp - start_temp) * epoch / warmup_epochs
    else:
        return final_temp
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, bottleneck_dim=256, out_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        weight_scale = self.last_layer.parametrizations.weight.original0
        weight_scale.data.fill_(1.0)
        weight_scale.requires_grad = False

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1)
        return self.last_layer(x)

class SegmentationSHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, final_size=(256, 256)):
        super().__init__()
        self.final_size = final_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    def forward(self, x):
        x = self.head(x)  # Shape: (B, 1, 8, 8)
        x = F.interpolate(x, size=self.final_size, mode='bilinear', align_corners=False)
        return x

class SegmentationMHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, final_size=(256, 256)):
        super().__init__()
        self.final_size = final_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
    def forward(self, x):
        x = self.head(x)  # Shape: (B, 1, 8, 8)
        x = F.interpolate(x, size=self.final_size, mode='bilinear', align_corners=False)
        return x
