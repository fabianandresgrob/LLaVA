"""
Tests for SAE bottleneck encode-only vs encode/decode modes.

Uses the real BatchTopKSAE checkpoint. Run on a GPU node with:
    SAE_CHECKPOINT_PATH=/path/to/ae.pt python -m pytest tests/test_sae_bottleneck.py -v

The SAE has activation_dim=1024, dict_size=8192, k=20.
"""

import os
import pytest
import torch
from types import SimpleNamespace

SAE_PATH = os.environ.get("SAE_CHECKPOINT_PATH")
SKIP_MSG = "SAE_CHECKPOINT_PATH not set"

# Real CLIP ViT-L/14 dimensions
CLIP_HIDDEN = 1024
SAE_DICT_SIZE = 8192
LLM_HIDDEN = 4096
NUM_PATCHES = 576  # 24x24 for 336px


@pytest.fixture
def clip_features():
    """Fake CLIP patch features with realistic shape."""
    return torch.randn(2, NUM_PATCHES, CLIP_HIDDEN)


# ---------------------------------------------------------------------------
# SAEBottleneck unit tests
# ---------------------------------------------------------------------------
class TestSAEBottleneck:
    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_encode_decode_output_shape(self, clip_features):
        from llava.model.sae_bottleneck import SAEBottleneck
        bottleneck = SAEBottleneck(SAE_PATH, encode_only=False, log_stats=False)
        out = bottleneck(clip_features)
        assert out.shape == (2, NUM_PATCHES, CLIP_HIDDEN)

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_encode_only_output_shape(self, clip_features):
        from llava.model.sae_bottleneck import SAEBottleneck
        bottleneck = SAEBottleneck(SAE_PATH, encode_only=True, log_stats=False)
        out = bottleneck(clip_features)
        assert out.shape == (2, NUM_PATCHES, SAE_DICT_SIZE)

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_output_dim_property(self):
        from llava.model.sae_bottleneck import SAEBottleneck
        enc_dec = SAEBottleneck(SAE_PATH, encode_only=False, log_stats=False)
        assert enc_dec.output_dim == CLIP_HIDDEN

        enc_only = SAEBottleneck(SAE_PATH, encode_only=True, log_stats=False)
        assert enc_only.output_dim == SAE_DICT_SIZE

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_encode_only_output_is_sparse_and_nonneg(self, clip_features):
        from llava.model.sae_bottleneck import SAEBottleneck
        bottleneck = SAEBottleneck(SAE_PATH, encode_only=True, log_stats=False)
        out = bottleneck(clip_features)
        assert (out >= 0).all(), "SAE encoder output should be non-negative (post-ReLU)"
        sparsity = (out == 0).float().mean()
        assert sparsity > 0.9, f"Expected highly sparse output, got {sparsity:.1%} zeros"

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_sae_params_frozen(self):
        from llava.model.sae_bottleneck import SAEBottleneck
        bottleneck = SAEBottleneck(SAE_PATH, encode_only=True, log_stats=False)
        for param in bottleneck.sae.parameters():
            assert not param.requires_grad

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_backward_compat_default(self):
        from llava.model.sae_bottleneck import SAEBottleneck
        bottleneck = SAEBottleneck(SAE_PATH, log_stats=False)  # no encode_only arg
        assert not bottleneck.encode_only
        assert bottleneck.output_dim == CLIP_HIDDEN


# ---------------------------------------------------------------------------
# Projector dimension tests (no SAE/GPU needed)
# ---------------------------------------------------------------------------
class TestProjectorDimensions:
    def test_projector_1024_input(self):
        """Standard encode/decode: projector takes CLIP hidden dim."""
        from llava.model.multimodal_projector.builder import build_vision_projector
        config = SimpleNamespace(mm_projector_type='mlp2x_gelu', mm_hidden_size=CLIP_HIDDEN, hidden_size=LLM_HIDDEN)
        proj = build_vision_projector(config)
        out = proj(torch.randn(2, NUM_PATCHES, CLIP_HIDDEN))
        assert out.shape == (2, NUM_PATCHES, LLM_HIDDEN)

    def test_projector_8192_input(self):
        """Encode-only: projector takes SAE dict_size dim."""
        from llava.model.multimodal_projector.builder import build_vision_projector
        config = SimpleNamespace(mm_projector_type='mlp2x_gelu', mm_hidden_size=SAE_DICT_SIZE, hidden_size=LLM_HIDDEN)
        proj = build_vision_projector(config)
        out = proj(torch.randn(2, NUM_PATCHES, SAE_DICT_SIZE))
        assert out.shape == (2, NUM_PATCHES, LLM_HIDDEN)


# ---------------------------------------------------------------------------
# End-to-end: SAE bottleneck → projector pipeline
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_encode_decode_pipeline(self, clip_features):
        """CLIP (1024d) → SAE enc/dec (1024d) → projector → LLM (4096d)."""
        from llava.model.sae_bottleneck import SAEBottleneck
        from llava.model.multimodal_projector.builder import build_vision_projector

        bottleneck = SAEBottleneck(SAE_PATH, encode_only=False, log_stats=False)
        config = SimpleNamespace(mm_projector_type='mlp2x_gelu', mm_hidden_size=bottleneck.output_dim, hidden_size=LLM_HIDDEN)
        projector = build_vision_projector(config)

        sae_out = bottleneck(clip_features)
        assert sae_out.shape == (2, NUM_PATCHES, CLIP_HIDDEN)
        llm_input = projector(sae_out)
        assert llm_input.shape == (2, NUM_PATCHES, LLM_HIDDEN)

    @pytest.mark.skipif(SAE_PATH is None, reason=SKIP_MSG)
    def test_encode_only_pipeline(self, clip_features):
        """CLIP (1024d) → SAE encode (8192d) → projector → LLM (4096d)."""
        from llava.model.sae_bottleneck import SAEBottleneck
        from llava.model.multimodal_projector.builder import build_vision_projector

        bottleneck = SAEBottleneck(SAE_PATH, encode_only=True, log_stats=False)
        config = SimpleNamespace(mm_projector_type='mlp2x_gelu', mm_hidden_size=bottleneck.output_dim, hidden_size=LLM_HIDDEN)
        projector = build_vision_projector(config)

        sae_out = bottleneck(clip_features)
        assert sae_out.shape == (2, NUM_PATCHES, SAE_DICT_SIZE)
        llm_input = projector(sae_out)
        assert llm_input.shape == (2, NUM_PATCHES, LLM_HIDDEN)
