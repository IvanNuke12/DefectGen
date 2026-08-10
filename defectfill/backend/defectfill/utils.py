import torch
import torch.nn.functional as F
import os
from typing import Optional, Dict, Any

def save_checkpoint(model, optimizer, step, path):
    """
    Save model checkpoint - Includes LoRA weights and learnable embeddings (Textual Inversion)
    
    Args:
        model: The DefectFill model instance
        optimizer: Optimizer state
        step: Current training step
        path: File path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract LoRA weights specifically from the UNet and Text Encoder
    checkpoint = {
        "step": step,
        "lora_rank": getattr(model, "lora_rank", None),
        "lora_alpha": getattr(model, "lora_alpha", None),
        "text_encoder_lora": {k: v for k, v in model.pipeline.text_encoder.state_dict().items() if "lora" in k},
        "unet_lora": {k: v for k, v in model.pipeline.unet.state_dict().items() if "lora" in k},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    
    # Save the learnable <defect> embedding (Textual Inversion component)
    if hasattr(model, 'placeholder_token_id'):
        token_embeds = model.pipeline.text_encoder.get_input_embeddings().weight.data
        checkpoint["learned_embedding"] = token_embeds[model.placeholder_token_id].clone()
        checkpoint["placeholder_token"] = model.placeholder_token
        checkpoint["placeholder_token_id"] = model.placeholder_token_id
        print(f"[Checkpoint] Saving learnable embedding: {model.placeholder_token} (id={model.placeholder_token_id})")
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def read_checkpoint_config(path):
    """
    Reads checkpoint metadata (step, lora_rank, lora_alpha) without loading weights.

    For checkpoints created before lora_rank/lora_alpha were saved (e.g. trained with
    an older version), the rank is inferred from the shape of the saved LoRA matrices
    (lora_A is [r, in_features], so shape[0] is the rank).

    Returns:
        dict with keys: step, lora_rank, lora_alpha (or None values when unknown)
    """
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location='cpu')
    cfg = {"step": ckpt.get("step", 0)}
    rank = ckpt.get("lora_rank")
    alpha = ckpt.get("lora_alpha")

    if rank is None:
        for group in ("text_encoder_lora", "unet_lora"):
            for k, v in ckpt.get(group, {}).items():
                if k.endswith("lora_A.default.weight") and v.dim() == 2:
                    rank = v.shape[0]
                    break
            if rank is not None:
                break

    cfg["lora_rank"] = rank
    cfg["lora_alpha"] = alpha
    return cfg

def load_checkpoint(model, optimizer, path) -> int:
    """
    Load model checkpoint - Restores LoRA weights and learnable embeddings
    
    Args:
        model: The DefectFill model instance
        optimizer: Optimizer to load state into
        path: Path to the checkpoint file
        
    Returns:
        Current step retrieved from the checkpoint
    """
    # Check if checkpoint exists
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found, starting from scratch")
        return 0
    
    # Load checkpoint to CPU first to avoid VRAM spikes
    checkpoint = torch.load(path, map_location='cpu')

    # Load text encoder LoRA weights
    text_encoder_sd = model.pipeline.text_encoder.state_dict()
    unet_sd = model.pipeline.unet.state_dict()

    # Verify LoRA shape compatibility (e.g. lora_rank mismatch between training and load)
    for group, model_sd in (("text_encoder_lora", text_encoder_sd), ("unet_lora", unet_sd)):
        for k, v in checkpoint[group].items():
            if k in model_sd and model_sd[k].shape != v.shape:
                raise ValueError(
                    f"Shape mismatch in '{k}': checkpoint has {tuple(v.shape)}, "
                    f"current model has {tuple(model_sd[k].shape)}. The checkpoint was "
                    f"trained with a different lora_rank (checkpoint rank {v.shape[0]} vs "
                    f"model rank {model_sd[k].shape[0]}). Load it with the matching "
                    f"--lora_rank/--lora_alpha, or use inference.py which auto-detects "
                    f"the rank from the checkpoint."
                )

    for k, v in checkpoint["text_encoder_lora"].items():
        if k in text_encoder_sd:
            text_encoder_sd[k] = v.to(text_encoder_sd[k].device)
    model.pipeline.text_encoder.load_state_dict(text_encoder_sd)
    
    # Load UNet LoRA weights
    for k, v in checkpoint["unet_lora"].items():
        if k in unet_sd:
            unet_sd[k] = v.to(unet_sd[k].device)
    model.pipeline.unet.load_state_dict(unet_sd)
    
    # Load the learnable <defect> embedding (Textual Inversion)
    if "learned_embedding" in checkpoint and hasattr(model, 'placeholder_token_id'):
        learned_emb = checkpoint["learned_embedding"]
        token_embeds = model.pipeline.text_encoder.get_input_embeddings().weight.data
        token_embeds[model.placeholder_token_id] = learned_emb.to(token_embeds.device)
        print(f"[Checkpoint] Loaded learnable embedding: {model.placeholder_token} (id={model.placeholder_token_id})")
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint["step"]


def compute_spatial_lpips(lpips_model, img1, img2, mask, smooth_boundary=True):
    """
    Calculates the Perceptual Distance specifically within the masked region using Spatial LPIPS
    
    Args:
        lpips_model: LPIPS model instance initialized with spatial=True
        img1: Reference image [B, 3, H, W], range [-1, 1]
        img2: Comparison image [B, 3, H, W], range [-1, 1]
        mask: Defect mask [B, 1, H, W], range [0, 1]
        smooth_boundary: Whether to blur the mask edges to avoid boundary artifacts
    
    Returns:
        lpips_score: LPIPS score for the masked region (scalar)
    """
    # 1. Compute spatial LPIPS map (pixel-wise perceptual distance)
    lpips_map = lpips_model(img1, img2)  # Output shape: [B, 1, H', W']
    
    # 2. Resize the mask to match the LPIPS output resolution
    mask_resized = F.interpolate(
        mask, 
        size=lpips_map.shape[-2:], 
        mode='bilinear',
        align_corners=False
    )
    
    # 3. Optional: Smooth boundary edges (Gaussian-like blur via AvgPool)
    if smooth_boundary:
        mask_smoothed = F.avg_pool2d(
            F.pad(mask_resized, (2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )
    else:
        mask_smoothed = mask_resized
    
    # 4. Mask-weighted summation
    weighted_sum = (lpips_map * mask_smoothed).sum(dim=(2, 3))
    mask_sum = mask_smoothed.sum(dim=(2, 3)) + 1e-8
    
    # 5. Return normalized LPIPS score
    return (weighted_sum / mask_sum).mean()


def compute_spatial_lpips_batch(lpips_model, reference, samples, mask, smooth_boundary=True):
    """
    Batch calculation of Spatial LPIPS - Evaluates all generated samples at once (FP16 compatible)
    
    This utilizes the "paired batch" mode of LPIPS by expanding the reference image
    to match the number of samples, allowing parallel evaluation in a single forward pass.
    
    Args:
        lpips_model: Spatial LPIPS model instance (spatial=True)
        reference: Single reference image [1, 3, H, W], range [-1, 1]
        samples: Multiple generated samples [N, 3, H, W], range [-1, 1]
        mask: Single defect mask [1, 1, H, W], range [0, 1]
        smooth_boundary: Enable mask boundary smoothing
    
    Returns:
        lpips_scores: A tensor of [N] LPIPS scores
    """
    num_samples = samples.shape[0]
    
    # LPIPS model expects FP32 input; cast based on model parameters
    lpips_dtype = next(lpips_model.parameters()).dtype
    
    # Expand reference and mask to match sample count N
    reference_expanded = reference.repeat(num_samples, 1, 1, 1).to(dtype=lpips_dtype)
    samples_for_lpips = samples.to(dtype=lpips_dtype)
    mask_expanded = mask.repeat(num_samples, 1, 1, 1)
    
    # Parallel forward pass for all N pairs
    lpips_maps = lpips_model(reference_expanded, samples_for_lpips)  # [N, 1, H', W']
    
    # Match mask size to LPIPS feature map resolution
    mask_resized = F.interpolate(
        mask_expanded.to(dtype=lpips_maps.dtype), 
        size=lpips_maps.shape[-2:], 
        mode='bilinear',
        align_corners=False
    )
    
    # Boundary smoothing
    if smooth_boundary:
        mask_smoothed = F.avg_pool2d(
            F.pad(mask_resized, (2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )
    else:
        mask_smoothed = mask_resized
    
    # Calculate weighted scores per sample
    weighted_sum = (lpips_maps * mask_smoothed).sum(dim=(2, 3))  # [N, 1]
    mask_sum = mask_smoothed.sum(dim=(2, 3)) + 1e-8             # [N, 1]
    
    lpips_scores = (weighted_sum / mask_sum).squeeze(1)  # [N]
    
    return lpips_scores