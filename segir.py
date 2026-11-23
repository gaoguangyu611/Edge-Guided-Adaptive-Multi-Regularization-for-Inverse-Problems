"""SEGIR: single-image self-supervised CT reconstruction demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, resize
from torch import Tensor, nn
from torch.nn import functional as functional
from tqdm import trange


LOGGER = logging.getLogger("segir")
LOSS_NAMES = ("data", "satv", "l1", "l2")


@dataclass(frozen=True)
class SEGIRConfig:
    """Parameters for a single phantom reconstruction experiment."""

    image_size: int = 256
    num_views: int = 180
    noise_level: float = 0.01
    iterations: int = 600
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    base_channels: int = 64
    seed: int = 42
    device: str = "auto"
    early_stop_after: int = 300
    early_stop_threshold: float = 1e-3
    output_dir: Path = Path("outputs/segir")

    def validate(self) -> None:
        """Reject unsupported geometry and invalid optimization settings."""
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16 for the three pooling stages.")
        if self.num_views < 1 or self.iterations < 1 or self.base_channels < 1:
            raise ValueError("num_views, iterations, and base_channels must be positive.")
        nonnegative = (
            self.noise_level,
            self.min_learning_rate,
            self.weight_decay,
            self.early_stop_threshold,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("Noise, weight decay, minimum LR, and threshold must be finite and nonnegative.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be finite and positive.")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate.")
        if self.early_stop_after < 0:
            raise ValueError("early_stop_after must be nonnegative.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32).")


def seed_everything(seed: int) -> None:
    """Seed the experiment; identical results across platforms are not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    """Select CPU or CUDA; scikit-image projections always execute on CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    return torch.device(name)


class ApproximateRadon(torch.autograd.Function):
    """CPU Radon forward with the original approximate backprojection backward.

    Unfiltered iradon is NOT the exact discrete transpose of skimage.radon.
    This preserves the supplied research script's first-order update rule;
    it is not an exact-gradient or higher-order differentiable projector.
    """

    @staticmethod
    def forward(ctx, image: Tensor, angles: np.ndarray) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 1 or image.shape[2] != image.shape[3]:
            raise ValueError("Expected image shape [batch, 1, size, size].")
        if image.dtype not in {torch.float32, torch.float64}:
            raise ValueError("The Radon wrapper supports float32 and float64 only.")
        theta = np.asarray(angles, dtype=np.float32)
        if theta.ndim != 1 or theta.size == 0 or not np.isfinite(theta).all():
            raise ValueError("angles must be a nonempty, finite 1-D array in degrees.")
        ctx.angles = theta.copy()
        ctx.image_size = image.shape[-1]
        image_array = image.detach().cpu().numpy()
        projections = [
            radon(sample[0], theta=ctx.angles, circle=True).T
            for sample in image_array
        ]
        return torch.from_numpy(np.stack(projections)[:, None]).to(
            device=image.device, dtype=image.dtype
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_sinogram: Tensor) -> tuple[Tensor, None]:
        gradient_array = grad_sinogram.detach().cpu().numpy()
        backprojections = [
            iradon(
                sample[0].T,
                theta=ctx.angles,
                filter_name=None,
                output_size=ctx.image_size,
                circle=True,
            )
            for sample in gradient_array
        ]
        gradient = torch.from_numpy(np.stack(backprojections)[:, None]).to(
            device=grad_sinogram.device, dtype=grad_sinogram.dtype
        )
        return gradient, None


def radon_torch(image: Tensor, angles: np.ndarray) -> Tensor:
    """Project [B, 1, H, W] to [B, 1, views, detectors]."""
    return ApproximateRadon.apply(image, angles)


def create_phantom(image_size: int) -> np.ndarray:
    """Resize and normalize the built-in Shepp-Logan phantom to [0, 1]."""
    phantom = resize(
        shepp_logan_phantom(),
        (image_size, image_size),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    phantom -= phantom.min()
    phantom /= max(float(phantom.max()), np.finfo(np.float32).eps)
    return phantom


def generate_sinogram(
    phantom: np.ndarray, num_views: int, noise_level: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise with std = noise_level * max(clean sinogram)."""
    angles = np.linspace(0.0, 180.0, num_views, endpoint=False, dtype=np.float32)
    clean_sinogram = radon(phantom, theta=angles, circle=True)
    generator = np.random.default_rng(seed)
    noise = generator.normal(
        0.0, noise_level * float(clean_sinogram.max()), clean_sinogram.shape
    )
    return (clean_sinogram + noise).astype(np.float32), angles


def fbp_reconstruction(
    sinogram: np.ndarray, angles: np.ndarray, image_size: int
) -> np.ndarray:
    """Compute nonnegative ramp-filtered backprojection on the target grid."""
    reconstruction = iradon(
        sinogram, theta=angles, output_size=image_size, filter_name="ramp", circle=True
    )
    return np.clip(reconstruction, 0.0, None).astype(np.float32)


class DoubleConv(nn.Module):
    """Two convolution, batch normalization, and ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class Down(nn.Module):
    """Halve spatial resolution, then apply a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class Up(nn.Module):
    """Upsample and combine encoder features through a skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, features: Tensor, skip: Tensor) -> Tensor:
        features = self.upsample(features)
        height_gap = skip.shape[2] - features.shape[2]
        width_gap = skip.shape[3] - features.shape[3]
        features = functional.pad(
            features,
            [width_gap // 2, width_gap - width_gap // 2,
             height_gap // 2, height_gap - height_gap // 2],
        )
        return self.conv(torch.cat((skip, features), dim=1))


class EdgePredictor(nn.Module):
    """Predict a learned soft weighting map, not a supervised edge label."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class SEGIR(nn.Module):
    """Three-level U-Net reconstruction branch with an edge-weight branch."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive.")
        self.edge_predictor = EdgePredictor(in_channels)
        self.input_conv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        edge = self.edge_predictor(image)
        first = self.input_conv(image)
        second = self.down1(first)
        third = self.down2(second)
        bottleneck = self.down3(third)
        features = self.up1(bottleneck, third)
        features = self.up2(features, second)
        features = self.up3(features, first)
        return self.output_conv(features), edge


class SEGIRLoss(nn.Module):
    """Data consistency and edge-adaptive priors with learned log weights."""

    def __init__(self) -> None:
        super().__init__()
        self.log_var = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in LOSS_NAMES}
        )

    @staticmethod
    def satv_loss(image: Tensor, edge: Tensor) -> Tensor:
        """Blend absolute and squared finite differences using soft weights."""
        height_gradient = image[:, :, 1:, :] - image[:, :, :-1, :]
        width_gradient = image[:, :, :, 1:] - image[:, :, :, :-1]
        height_edge = functional.interpolate(
            edge, size=height_gradient.shape[2:], mode="bilinear", align_corners=False
        )
        width_edge = functional.interpolate(
            edge, size=width_gradient.shape[2:], mode="bilinear", align_corners=False
        )
        return (
            (height_gradient.abs() * (1 - height_edge)).mean()
            + (width_gradient.abs() * (1 - width_edge)).mean()
            + (height_gradient.square() * height_edge).mean()
            + (width_gradient.square() * width_edge).mean()
        )

    def forward(
        self, output: Tensor, edge: Tensor, observation: Tensor, angles: np.ndarray
    ) -> tuple[Tensor, dict[str, float]]:
        prediction = radon_torch(output, angles)
        if prediction.shape != observation.shape:
            raise ValueError(
                f"Sinogram shape mismatch: predicted {tuple(prediction.shape)}, "
                f"observed {tuple(observation.shape)}. Check angles and detector geometry; "
                "observations are never silently resized."
            )
        losses = {
            "data": functional.mse_loss(prediction, observation),
            "satv": self.satv_loss(output, edge),
            "l1": (output.abs() * (1 - edge)).mean(),
            "l2": (output.square() * edge).mean(),
        }
        total = torch.stack([
            torch.exp(-self.log_var[name]) * loss + self.log_var[name]
            for name, loss in losses.items()
        ]).sum()
        values = {name: float(loss.detach().item()) for name, loss in losses.items()}
        values["total"] = float(total.detach().item())
        return total, values


def compute_psnr(reference: np.ndarray, reconstruction: np.ndarray) -> float | None:
    """Return PSNR for data_range=1; None denotes a perfect reconstruction."""
    error = reference.astype(np.float64) - reconstruction.astype(np.float64)
    mse = float(np.mean(error**2))
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else None


def save_figure(
    phantom: np.ndarray, fbp: np.ndarray, reconstruction: np.ndarray,
    edge: np.ndarray, destination: Path,
) -> None:
    """Save a headless comparison with a shared reconstruction display scale."""
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(
        axes[:3], (phantom, fbp, reconstruction),
        ("Shepp-Logan phantom", "FBP initialization", "SEGIR reconstruction"),
    ):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
    axes[3].imshow(edge, cmap="hot", vmin=0.0, vmax=1.0)
    axes[3].set_title("Learned edge weights")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run_experiment(config: SEGIRConfig) -> dict:
    """Optimize one noisy sinogram and save reproducibility artifacts."""
    config.validate()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path must be an empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    phantom = create_phantom(config.image_size)
    sinogram, angles = generate_sinogram(
        phantom, config.num_views, config.noise_level, config.seed
    )
    fbp = fbp_reconstruction(sinogram, angles, config.image_size)
    initial_image = torch.from_numpy(fbp.copy())[None, None].to(device)
    observation = torch.from_numpy(sinogram.T.copy())[None, None].to(device)
    model = SEGIR(base_channels=config.base_channels).to(device)
    criterion = SEGIRLoss().to(device)
    parameters = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=config.min_learning_rate
    )
    LOGGER.info("Device: %s; model parameters: %s", device, f"{sum(parameter.numel() for parameter in model.parameters()):,}")
    LOGGER.warning("Radon backward uses approximate CPU backprojection, not an exact discrete adjoint.")
    history = {name: [] for name in ("total", *LOSS_NAMES)}
    stopping_threshold = config.early_stop_threshold * float(sinogram.max()) ** 2
    model.train()
    progress = trange(config.iterations, desc="SEGIR")
    for iteration in progress:
        optimizer.zero_grad(set_to_none=True)
        reconstruction, edge = model(initial_image)
        loss, values = criterion(reconstruction, edge, observation, angles)
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite loss detected; try a lower learning rate.")
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        for name in history:
            history[name].append(values[name])
        if iteration % 5 == 0:
            progress.set_postfix(data=f"{values['data']:.4f}", total=f"{values['total']:.4f}")
        if (
            config.early_stop_threshold > 0
            and iteration + 1 > config.early_stop_after
            and values["data"] < stopping_threshold
        ):
            LOGGER.info("Early stopping after %d iterations.", iteration + 1)
            break
    model.eval()
    with torch.no_grad():
        final_image, final_edge = model(initial_image)
    reconstruction = final_image[0, 0].cpu().numpy()
    edge = final_edge[0, 0].cpu().numpy()
    if not np.isfinite(reconstruction).all() or not np.isfinite(edge).all():
        raise FloatingPointError("Non-finite final reconstruction detected.")
    saved_config = asdict(config)
    saved_config["output_dir"] = str(output_dir)
    metrics = {
        "iterations_completed": len(history["total"]),
        "fbp_psnr_db": compute_psnr(phantom, fbp),
        "segir_psnr_db": compute_psnr(phantom, reconstruction),
        "psnr_data_range": 1.0,
        "adaptive_weights": {
            name: float(torch.exp(-value.detach()).item())
            for name, value in criterion.log_var.items()
        },
        "resolved_device": str(device),
        "versions": {
            "torch": str(torch.__version__), "numpy": np.__version__,
            "scikit-image": skimage.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    for filename, payload in (
        ("config.json", saved_config), ("history.json", history), ("metrics.json", metrics)
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    np.savez_compressed(
        output_dir / "reconstruction.npz", phantom=phantom, sinogram=sinogram,
        angles=angles, fbp=fbp, reconstruction=reconstruction, edge=edge,
    )
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "loss_state_dict": {name: value.detach().cpu() for name, value in criterion.state_dict().items()},
            "config": saved_config,
            "iterations_completed": metrics["iterations_completed"],
        },
        output_dir / "segir.pt",
    )
    save_figure(phantom, fbp, reconstruction, edge, output_dir / "segir_reconstruction.png")
    LOGGER.info("Results saved to %s", output_dir.resolve())
    LOGGER.info("PSNR (dB): FBP=%s, SEGIR=%s", metrics["fbp_psnr_db"], metrics["segir_psnr_db"])
    return metrics


def parse_args() -> SEGIRConfig:
    """Expose experiment settings without editing source code."""
    defaults = SEGIRConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image-size", type=int, default=defaults.image_size, help="Square image size (>=16).")
    parser.add_argument("--num-views", type=int, default=defaults.num_views, help="Views uniformly sampled over [0, 180) degrees.")
    parser.add_argument("--noise-level", type=float, default=defaults.noise_level, help="Gaussian noise std relative to peak clean projection.")
    parser.add_argument("--iterations", type=int, default=defaults.iterations, help="Maximum optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="AdamW learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate, help="Cosine scheduler minimum learning rate.")
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip, help="Combined model/loss gradient norm limit.")
    parser.add_argument("--base-channels", type=int, default=defaults.base_channels, help="First U-Net stage width.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device, help="Neural network device; projections remain on CPU.")
    parser.add_argument("--early-stop-after", type=int, default=defaults.early_stop_after, help="Minimum completed steps before early stopping is checked.")
    parser.add_argument("--early-stop-threshold", type=float, default=defaults.early_stop_threshold, help="Stop if data MSE < threshold * peak noisy projection squared; 0 disables.")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Output directory, which must be empty or absent.")
    return SEGIRConfig(**vars(parser.parse_args()))


def main() -> None:
    """Run the command-line reconstruction demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_experiment(parse_args())
    except (ValueError, FileExistsError, FloatingPointError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
