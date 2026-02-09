"""
Test script to verify SAE bottleneck integration into LLaVA.

Checks:
1. Model loads with SAE flag enabled
2. Forward pass produces correct output shapes
3. SAE parameters are frozen (requires_grad=False)
4. SAE parameters are NOT in optimizer parameter groups
5. Model without SAE flag behaves normally (baseline)
"""

import os
import sys
import torch
import argparse


def test_sae_bottleneck_standalone(sae_checkpoint_path: str):
    """Test the SAEBottleneck module in isolation."""
    print("=" * 60)
    print("Test 1: SAEBottleneck standalone")
    print("=" * 60)

    from llava.model.sae_bottleneck import SAEBottleneck

    bottleneck = SAEBottleneck(sae_checkpoint_path)
    print(f"  SAE loaded: {bottleneck}")

    # Test forward pass with dummy data
    B, N, D = 2, 576, 1024
    x = torch.randn(B, N, D)
    x_hat = bottleneck(x)

    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} != {x.shape}"
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {x_hat.shape}")

    # Check reconstruction quality
    recon_error = (x - x_hat).norm() / x.norm()
    print(f"  Relative reconstruction error: {recon_error.item():.4f}")

    # Check all params frozen
    for name, param in bottleneck.named_parameters():
        assert not param.requires_grad, f"Parameter {name} has requires_grad=True!"
    print("  All parameters frozen: PASS")

    # Test bf16 compatibility
    x_bf16 = x.to(torch.bfloat16)
    x_hat_bf16 = bottleneck(x_bf16)
    assert x_hat_bf16.dtype == torch.bfloat16, f"Output dtype {x_hat_bf16.dtype} != bf16"
    print("  bf16 compatibility: PASS")

    print("  PASSED\n")


def test_model_with_sae(sae_checkpoint_path: str):
    """Test LLaVA model with SAE bottleneck enabled."""
    print("=" * 60)
    print("Test 2: LLaVA model with SAE bottleneck")
    print("=" * 60)

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from dataclasses import dataclass, field
    from typing import Optional

    # Simulate model args
    @dataclass
    class MockModelArgs:
        vision_tower: str = "openai/clip-vit-large-patch14-336"
        mm_vision_select_layer: int = -2
        mm_vision_select_feature: str = "patch"
        pretrain_mm_mlp_adapter: Optional[str] = None
        mm_projector_type: str = "mlp2x_gelu"
        mm_use_im_start_end: bool = False
        mm_use_im_patch_token: bool = False
        mm_patch_merge_type: str = "flat"
        use_sae_bottleneck: bool = True
        sae_checkpoint_path: str = sae_checkpoint_path
        tune_mm_mlp_adapter: bool = False

    # Build model from scratch
    from transformers import LlamaConfig
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

    config = LlavaConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
    config.mm_vision_tower = "openai/clip-vit-large-patch14-336"
    config.mm_projector_type = "mlp2x_gelu"
    config.mm_hidden_size = 1024
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"

    print("  Loading model (this may take a moment)...")
    model = LlavaLlamaForCausalLM(config)

    # Initialize vision modules with SAE
    model_args = MockModelArgs()
    model.get_model().initialize_vision_modules(model_args=model_args)

    # Verify SAE is attached
    sae_module = model.get_model().sae_bottleneck
    assert sae_module is not None, "SAE bottleneck not found on model!"
    print(f"  SAE bottleneck attached: {sae_module}")

    # Check SAE params are frozen
    sae_param_count = 0
    for name, param in model.named_parameters():
        if "sae_bottleneck" in name:
            assert not param.requires_grad, f"SAE param {name} has requires_grad=True!"
            sae_param_count += param.numel()
    print(f"  SAE parameters: {sae_param_count:,} (all frozen)")

    # Check trainable params don't include SAE
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    sae_trainable = [n for n in trainable_params if "sae_bottleneck" in n]
    assert len(sae_trainable) == 0, f"SAE params in trainable list: {sae_trainable}"
    print("  SAE excluded from trainable params: PASS")

    # Test encode_images
    print("  Testing encode_images forward pass...")
    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    dummy_image = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        image_features = model.encode_images(dummy_image)
    print(f"  encode_images output shape: {image_features.shape}")
    assert image_features.shape[0] == 1
    assert image_features.shape[1] == 576  # 24*24 patches
    assert image_features.shape[2] == config.hidden_size
    print("  encode_images shape: PASS")

    print("  PASSED\n")


def test_model_without_sae():
    """Test that model without SAE flag is unchanged (baseline behavior)."""
    print("=" * 60)
    print("Test 3: LLaVA model WITHOUT SAE (baseline)")
    print("=" * 60)

    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class MockModelArgs:
        vision_tower: str = "openai/clip-vit-large-patch14-336"
        mm_vision_select_layer: int = -2
        mm_vision_select_feature: str = "patch"
        pretrain_mm_mlp_adapter: Optional[str] = None
        mm_projector_type: str = "mlp2x_gelu"
        mm_use_im_start_end: bool = False
        mm_use_im_patch_token: bool = False
        mm_patch_merge_type: str = "flat"
        use_sae_bottleneck: bool = False
        sae_checkpoint_path: Optional[str] = None
        tune_mm_mlp_adapter: bool = False

    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

    config = LlavaConfig.from_pretrained("lmsys/vicuna-7b-v1.5")
    config.mm_vision_tower = "openai/clip-vit-large-patch14-336"
    config.mm_projector_type = "mlp2x_gelu"
    config.mm_hidden_size = 1024
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"

    print("  Loading model...")
    model = LlavaLlamaForCausalLM(config)

    model_args = MockModelArgs()
    model.get_model().initialize_vision_modules(model_args=model_args)

    # Verify SAE is NOT attached
    sae_module = getattr(model.get_model(), 'sae_bottleneck', None)
    assert sae_module is None, "SAE bottleneck should be None in baseline mode!"
    print("  SAE bottleneck is None: PASS")

    print("  PASSED\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SAE integration into LLaVA")
    parser.add_argument(
        "--sae_checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained SAE checkpoint (directory or .pt file)"
    )
    parser.add_argument(
        "--skip_model_test",
        action="store_true",
        help="Skip full model tests (only run standalone SAE test)"
    )
    args = parser.parse_args()

    print("\nSAE Integration Tests")
    print("=" * 60)
    print(f"SAE checkpoint: {args.sae_checkpoint_path}\n")

    # Test 1: Standalone SAE
    test_sae_bottleneck_standalone(args.sae_checkpoint_path)

    if not args.skip_model_test:
        # Test 2: Model with SAE
        test_model_with_sae(args.sae_checkpoint_path)

        # Test 3: Model without SAE (baseline)
        test_model_without_sae()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
