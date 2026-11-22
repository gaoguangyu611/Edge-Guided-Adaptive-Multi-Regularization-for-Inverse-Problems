"""
SEGIR: Self-supervised Edge-Guided Iterative Reconstruction for CT Images
=========================================================================
A PyTorch implementation of self-supervised CT reconstruction using
differentiable Radon transform and Bayesian adaptive loss weighting.

Author: [Tianyi Ye]
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import radon as sk_radon, iradon as sk_iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import resize as sk_resize
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================= Configuration =================

@dataclass
class SEGIRConfig:
    """Configuration for SEGIR model and training."""
    # Data parameters
    image_size: int = 256
    n_views: int = 180
    noise_level: float = 0.01
    angle_range: float = 180.0

    # Model parameters
    in_channels: int = 1
    out_channels: int = 1

    # Training parameters
    n_iterations: int = 600
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eta_min: float = 1e-5
    grad_clip: float = 1.0
    early_stop_iter: int = 300
    early_stop_threshold: float = 0.001

    # Output parameters
    save_path: str = 'segir_shepp_logan_bayesian.png'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= Differentiable Operators =================

class DiffRadon(torch.autograd.Function):
    """Differentiable Radon transform using skimage."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, theta_arr: np.ndarray, size: int) -> torch.Tensor:
        ctx.theta = theta_arr
        ctx.size = size
        x_np = x.detach().cpu().numpy()
        out = [sk_radon(x_np[b, 0], theta=ctx.theta, circle=True).T
               for b in range(x_np.shape[0])]
        return torch.from_numpy(np.stack(out)[:, None, :, :]).to(x.device).float()

    @staticmethod
    def backward(ctx, grad_sino: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        gs_np = grad_sino.detach().cpu().numpy()
        out = [sk_iradon(gs_np[b, 0].T, theta=ctx.theta, filter_name=None,
                         output_size=ctx.size) for b in range(gs_np.shape[0])]
        return torch.from_numpy(np.stack(out)[:, None, :, :]).to(grad_sino.device).float(), None, None


class DiffIRadon(torch.autograd.Function):
    """Differentiable Inverse Radon transform using skimage."""

    @staticmethod
    def forward(ctx, sino: torch.Tensor, theta_arr: np.ndarray, size: int) -> torch.Tensor:
        ctx.theta = theta_arr
        ctx.size = size
        s_np = sino.detach().cpu().numpy()
        out = [sk_iradon(s_np[b, 0].T, theta=ctx.theta, filter_name=None,
                         output_size=size) for b in range(s_np.shape[0])]
        return torch.from_numpy(np.stack(out)[:, None, :, :]).to(sino.device).float()

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        gx_np = grad_x.detach().cpu().numpy()
        out = [sk_radon(gx_np[b, 0], theta=ctx.theta, circle=True).T
               for b in range(gx_np.shape[0])]
        return torch.from_numpy(np.stack(out)[:, None, :, :]).to(grad_x.device).float(), None, None


def radon_torch(x: torch.Tensor, theta_arr: np.ndarray, size: int) -> torch.Tensor:
    """Wrapper for differentiable Radon transform."""
    return DiffRadon.apply(x, theta_arr, size)


def iradon_torch(sino: torch.Tensor, theta_arr: np.ndarray, size: int) -> torch.Tensor:
    """Wrapper for differentiable inverse Radon transform."""
    return DiffIRadon.apply(sino, theta_arr, size)


# ================= Data Generation =================

def generate_180deg_sinogram(phantom: np.ndarray,
                             n_views: int = 180,
                             noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 180-degree sinogram from phantom image.

    Args:
        phantom: Input phantom image
        n_views: Number of projection views
        noise_level: Gaussian noise level relative to max sinogram value

    Returns:
        sinogram_noisy: Noisy sinogram
        angles: Projection angles in degrees
    """
    angles = np.linspace(0, 180, n_views, endpoint=False).astype(np.float32)
    sinogram = sk_radon(phantom, theta=angles, circle=True)
    noise = np.random.normal(0, noise_level * np.max(sinogram), sinogram.shape)
    sinogram_noisy = sinogram + noise
    return sinogram_noisy, angles


def fbp_reconstruction(sinogram: np.ndarray,
                       angles: np.ndarray,
                       size: int = 256) -> np.ndarray:
    """
    Filtered Back Projection reconstruction.

    Args:
        sinogram: Input sinogram
        angles: Projection angles
        size: Output image size

    Returns:
        fbp: Reconstructed image
    """
    fbp = sk_iradon(sinogram, theta=angles, circle=True, filter_name='ramp')
    if fbp.shape[0] != size:
        fbp = sk_resize(fbp, (size, size), preserve_range=True)
    return np.clip(fbp, 0, None).astype(np.float32)


# ================= Network Architecture =================

class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block with max pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block with transpose convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle size mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class EdgePredictor(nn.Module):
    """Predicts edge map from input image."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SEGIR_Net(nn.Module):
    """
    SEGIR Network: Self-supervised Edge-Guided Iterative Reconstruction.

    U-Net based architecture with edge prediction branch.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.edge_predictor = EdgePredictor(in_channels)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge = self.edge_predictor(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x), edge


# ================= Loss Functions =================

class BayesianAdaptiveSEGIRLoss(nn.Module):
    """
    Bayesian Adaptive Loss for SEGIR with learnable loss weights.

    Combines data consistency, spatially adaptive TV, L1, and L2 losses.
    """

    def __init__(self):
        super().__init__()
        self.log_var = nn.ParameterDict({
            'data': nn.Parameter(torch.zeros(1)),
            'satv': nn.Parameter(torch.zeros(1)),
            'l1': nn.Parameter(torch.zeros(1)),
            'l2': nn.Parameter(torch.zeros(1)),
        })

    def _satv_loss(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Spatially Adaptive Total Variation loss."""
        grad_x_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        E_h = F.interpolate(edge_weight, size=grad_x_h.shape[2:],
                            mode='bilinear', align_corners=False)
        E_w = F.interpolate(edge_weight, size=grad_x_w.shape[2:],
                            mode='bilinear', align_corners=False)
        return (torch.abs(grad_x_h) * (1 - E_h)).mean() + \
            (torch.abs(grad_x_w) * (1 - E_w)).mean() + \
            ((grad_x_h ** 2) * E_h).mean() + \
            ((grad_x_w ** 2) * E_w).mean()

    def _adaptive_l1(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Edge-adaptive L1 loss."""
        E = F.interpolate(edge_weight, size=x.shape[2:],
                          mode='bilinear', align_corners=False)
        return (torch.abs(x) * (1 - E)).mean()

    def _adaptive_l2(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Edge-adaptive L2 loss."""
        E = F.interpolate(edge_weight, size=x.shape[2:],
                          mode='bilinear', align_corners=False)
        return ((x ** 2) * E).mean()

    def forward(self, output: torch.Tensor, edge: torch.Tensor,
                y_obs: torch.Tensor, theta_arr: np.ndarray, size: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        sino_pred = radon_torch(output, theta_arr, size)

        # Handle shape mismatch
        if sino_pred.shape != y_obs.shape:
            logger.warning(f"Shape mismatch: sino_pred {sino_pred.shape} vs y_obs {y_obs.shape}")
            y_obs = F.interpolate(y_obs, size=sino_pred.shape[2:], mode='bilinear', align_corners=False)

        data_loss = F.mse_loss(sino_pred, y_obs)
        satv_loss = self._satv_loss(output, edge)
        l1_loss = self._adaptive_l1(output, edge)
        l2_loss = self._adaptive_l2(output, edge)

        losses = {'data': data_loss, 'satv': satv_loss, 'l1': l1_loss, 'l2': l2_loss}

        # Bayesian weighted sum
        total_loss = torch.tensor(0.0, device=data_loss.device)
        for key in losses:
            total_loss += torch.exp(-self.log_var[key]) * losses[key] + self.log_var[key]

        loss_dict = {key: losses[key].item() for key in losses}
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ================= Training =================

def train_single_sample(config: SEGIRConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    """
    Zero-shot self-supervised training on a single sample.

    Args:
        config: SEGIR configuration

    Returns:
        final_output: Reconstructed image
        final_edge: Edge map
        loss_history: Training loss history
    """
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Load Shepp-Logan phantom
    logger.info("Loading Shepp-Logan phantom...")
    phantom = shepp_logan_phantom().astype(np.float32)
    if phantom.shape != (config.image_size, config.image_size):
        phantom = sk_resize(phantom, (config.image_size, config.image_size), preserve_range=True)
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())

    # Generate sinogram
    logger.info(f"Generating {config.angle_range}° sinogram with {config.n_views} views...")
    sinogram, angles = generate_180deg_sinogram(phantom, n_views=config.n_views,
                                                noise_level=config.noise_level)

    # FBP reconstruction
    logger.info("Performing FBP reconstruction...")
    fbp = fbp_reconstruction(sinogram, angles, size=config.image_size)

    # Convert to tensors
    x_init = torch.FloatTensor(fbp).unsqueeze(0).unsqueeze(0).to(device)
    y_obs = torch.FloatTensor(sinogram.T).unsqueeze(0).unsqueeze(0).to(device)
    theta_arr = angles.astype(np.float32)

    logger.info(f"y_obs shape: {y_obs.shape}")
    logger.info(f"x_init shape: {x_init.shape}")

    # Initialize model
    model = SEGIR_Net(in_channels=config.in_channels, out_channels=config.out_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss function
    criterion = BayesianAdaptiveSEGIRLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': config.lr},
        {'params': criterion.parameters(), 'lr': config.lr}
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_iterations, eta_min=config.eta_min
    )

    # Training loop
    logger.info("Starting zero-shot self-supervised training...")
    model.train()
    loss_history = {k: [] for k in ['total', 'data', 'satv', 'l1', 'l2']}

    pbar = tqdm(range(config.n_iterations), desc="Training")
    for iteration in pbar:
        optimizer.zero_grad()
        output, edge = model(x_init)
        loss, loss_dict = criterion(output, edge, y_obs, theta_arr, config.image_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        scheduler.step()

        for key in loss_history:
            loss_history[key].append(loss_dict[key])

        if iteration % 5 == 0:
            pbar.set_postfix({
                'data': f"{loss_dict['data']:.4f}",
                'satv': f"{loss_dict['satv']:.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
                'l2': f"{loss_dict['l2']:.4f}"
            })

        # Early stopping
        if iteration > config.early_stop_iter and \
                loss_dict['data'] < config.early_stop_threshold * (sinogram.max() ** 2):
            logger.info(f"Early stopping at iteration {iteration}")
            break

    # Final reconstruction
    model.eval()
    with torch.no_grad():
        final_output, final_edge = model(x_init)
        final_output = final_output.cpu().squeeze().numpy()
        final_edge = final_edge.cpu().squeeze().numpy()

    return final_output, final_edge, loss_history, phantom, fbp


# ================= Visualization =================

def visualize_results(phantom: np.ndarray, fbp: np.ndarray,
                      final_output: np.ndarray, final_edge: np.ndarray,
                      loss_history: Dict[str, List[float]],
                      save_path: str = 'segir_results.png'):
    """Visualize reconstruction results and training curves."""
    fig = plt.figure(figsize=(18, 6))

    # Row 1: Images
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(phantom, cmap='gray')
    ax1.set_title('Shepp-Logan Phantom\n(Ground Truth)')
    ax1.axis('off')

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(fbp, cmap='gray')
    ax2.set_title('FBP Init')
    ax2.axis('off')

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(final_output, cmap='gray')
    ax3.set_title('SEGIR Reconstruction')
    ax3.axis('off')

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(final_edge, cmap='hot')
    ax4.set_title('Edge Map')
    ax4.axis('off')

    # Row 2: Loss curves
    ax5 = plt.subplot(2, 1, 2)
    for key in ['total', 'data', 'satv', 'l1', 'l2']:
        ax5.plot(loss_history[key], label=key)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training Loss Curves')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Compute PSNR
    psnr = 10 * np.log10(1.0 / np.mean((final_output - phantom) ** 2))
    logger.info(f"PSNR: {psnr:.2f} dB")

    return psnr


# ================= Main =================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SEGIR: Self-supervised CT Reconstruction')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--n_views', type=int, default=180, help='Number of projection views')
    parser.add_argument('--noise_level', type=float, default=0.01, help='Noise level')
    parser.add_argument('--n_iterations', type=int, default=600, help='Training iterations')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='segir_results.png', help='Save path')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    config = SEGIRConfig(
        image_size=args.image_size,
        n_views=args.n_views,
        noise_level=args.noise_level,
        n_iterations=args.n_iterations,
        lr=args.lr,
        save_path=args.save_path
    )

    # Train
    final_output, final_edge, loss_history, phantom, fbp = train_single_sample(config)

    # Visualize
    psnr = visualize_results(phantom, fbp, final_output, final_edge, loss_history, config.save_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()