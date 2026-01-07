import torch
import torch.nn as nn
import math

from utils import *


class HighFrequencyGenerator(nn.Module):
    """
        DCT + High-Pass Filter + iDCT
    """
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # DCT
        X = dct_2d(x)

        z = build_hpf_mask(H, W, self.alpha, device)
        z = z.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        X_f = z * X

        # iDCT
        Fi = idct_2d(X_f)

        return Fi

class CP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1)
        )
        self.gmp = nn.AdaptiveMaxPool2d(1)

class CP(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)

        self.conv_fuse = nn.Conv2d(
            2 * (channels // reduction),
            channels,
            kernel_size=1,
            bias=False
        )

        self.act = nn.Sigmoid()

    def forward(self, Fi, Ci):

        gap = self.conv_gap(self.gap(Fi))
        gmp = self.conv_gmp(self.gmp(Fi))

        # s: pixel-by-pixel summation
        s = gap + gmp

        # c: channel-wise concatenation
        c = torch.cat([gap, gmp], dim=1)

        # channel weight
        u_cp = self.act(self.conv_fuse(c))

        # apply to Ci
        return Ci * u_cp






if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open("img.png").convert("L")
    transform = T.ToTensor()
    x = transform(img).unsqueeze(0).to(device)

    model = HighFrequencyGenerator(
        alpha=0.1
    ).to(device)

    with torch.no_grad():
        y = model(x)

    x_np = x[0, 0].cpu().numpy()
    y_np = y[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(x_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Filtered")
    plt.imshow(y_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("High-frequency (|Δ|)")
    plt.imshow(abs(y_np - x_np), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()