"""
SAE Bottleneck module for LLaVA.

Wraps a pretrained BatchTopK Sparse Autoencoder (SAE) as a frozen bottleneck
layer inserted between the CLIP vision encoder and the MLP projection.
The SAE encode/decode is applied to each token independently.
"""

import os
import sys
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAEBottleneck(nn.Module):
    """
    A frozen SAE bottleneck that applies encode then decode to visual features.

    The SAE was trained on patch token activations from CLIP ViT-L/14-336 layer 22
    using the BatchTopK architecture from the dictionary_learning library.

    At inference time, BatchTopK uses a learned threshold: ReLU(x - gamma)
    instead of the batch top-k selection used during SAE training.
    """

    def __init__(self, sae_checkpoint_path: str, log_stats: bool = True, log_interval: int = 100):
        super().__init__()
        self.log_stats = log_stats
        self.log_interval = log_interval
        self._step_counter = 0

        # Load the SAE
        self.sae = self._load_sae(sae_checkpoint_path)

        # Freeze all SAE parameters
        for param in self.sae.parameters():
            param.requires_grad = False
        # Also freeze buffers (k, threshold are registered as buffers)
        self.sae.eval()

        logger.info(
            f"SAE Bottleneck initialized: activation_dim={self.sae.activation_dim}, "
            f"dict_size={self.sae.dict_size}, k={self.sae.k.item()}, "
            f"threshold={self.sae.threshold.item():.6f}"
        )

    def _load_sae(self, checkpoint_path: str):
        """Load the BatchTopKSAE from the dictionary_learning library."""
        # Add the sae-for-vlm repo to sys.path so we can import dictionary_learning
        llava_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sae_repo_path = os.path.join(os.path.dirname(llava_root), "sae-for-vlm")
        if os.path.isdir(sae_repo_path) and sae_repo_path not in sys.path:
            sys.path.insert(0, sae_repo_path)

        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

        # Handle both directory path (look for ae.pt) and direct file path
        if os.path.isdir(checkpoint_path):
            ae_path = os.path.join(checkpoint_path, "ae.pt")
            if not os.path.exists(ae_path):
                raise FileNotFoundError(
                    f"Could not find ae.pt in {checkpoint_path}. "
                    f"Contents: {os.listdir(checkpoint_path)}"
                )
        else:
            ae_path = checkpoint_path

        logger.info(f"Loading SAE from {ae_path}")
        sae = BatchTopKSAE.from_pretrained(ae_path)
        return sae

    @torch.no_grad()
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Apply SAE encode/decode bottleneck to visual features.

        Args:
            image_features: [B, num_tokens, D] visual features from CLIP

        Returns:
            Reconstructed features of same shape [B, num_tokens, D]
        """
        input_dtype = image_features.dtype
        B, N, D = image_features.shape

        # Match input dtype to SAE weight dtype (ZeRO-3 may convert weights to bf16)
        sae_dtype = self.sae.encoder.weight.dtype
        x = image_features.to(sae_dtype)

        # Reshape to [B*N, D] for per-token SAE processing
        x_flat = x.reshape(B * N, D)

        # SAE encode/decode (uses threshold-based activation at test time)
        x_hat_flat, encoded_acts = self.sae(x_flat, output_features=True)

        # Log statistics periodically
        if self.log_stats and self.training:
            self._step_counter += 1
            if self._step_counter % self.log_interval == 0:
                self._log_sae_stats(x_flat, x_hat_flat, encoded_acts)

        # Reshape back to [B, N, D]
        x_hat = x_hat_flat.reshape(B, N, D)

        # Cast back to input dtype (bf16 during mixed precision training)
        return x_hat.to(input_dtype)

    def _log_sae_stats(self, x: torch.Tensor, x_hat: torch.Tensor, encoded_acts: torch.Tensor):
        """Log SAE statistics for wandb monitoring."""
        try:
            import wandb
            if not wandb.run:
                return

            # Reconstruction error: ||x - x_hat||_2 / ||x||_2
            recon_error = (x - x_hat).norm(dim=-1) / (x.norm(dim=-1) + 1e-8)
            mean_recon_error = recon_error.mean().item()

            # Sparsity: average number of active features per token
            active_per_token = (encoded_acts > 0).float().sum(dim=-1).mean().item()

            # Feature activation statistics
            active_mask = encoded_acts > 0
            if active_mask.any():
                active_values = encoded_acts[active_mask]
                mean_activation = active_values.mean().item()
                max_activation = active_values.max().item()
            else:
                mean_activation = 0.0
                max_activation = 0.0

            # FVE (Fraction of Variance Explained)
            x_centered = x - x.mean(dim=0, keepdim=True)
            total_var = x_centered.pow(2).sum()
            residual_var = (x - x_hat).pow(2).sum()
            fve = 1 - (residual_var / (total_var + 1e-8))

            wandb.log({
                "sae/reconstruction_error": mean_recon_error,
                "sae/active_features_per_token": active_per_token,
                "sae/mean_activation": mean_activation,
                "sae/max_activation": max_activation,
                "sae/fve": fve.item(),
            }, commit=False)
        except Exception:
            pass  # Don't let logging errors break training

    def extra_repr(self) -> str:
        return (
            f"activation_dim={self.sae.activation_dim}, "
            f"dict_size={self.sae.dict_size}, "
            f"k={self.sae.k.item()}, "
            f"threshold={self.sae.threshold.item():.6f}"
        )
