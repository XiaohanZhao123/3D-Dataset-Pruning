#!/usr/bin/env python
"""Standalone Point-MAE model loader script.

This script runs in isolation to avoid import conflicts with the main codebase.
It loads a Point-MAE PointTransformer model and saves it in a pickle format
that can be loaded by the main codebase.

Usage:
    python _pointmae_loader.py <checkpoint_path> <output_path> [--device <device>]
"""

import argparse
import sys
from pathlib import Path

# Add Point-MAE to path FIRST before any other imports
POINTMAE_PATH = Path(__file__).parent.parent / "third_party" / "pointmae"
sys.path.insert(0, str(POINTMAE_PATH))


def main():
    parser = argparse.ArgumentParser(description="Load Point-MAE model")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("output", help="Path to save loaded model state dict")
    parser.add_argument("--device", default="cpu", help="Device to load model on")
    parser.add_argument("--return-config", action="store_true",
                        help="Print config as JSON and exit")
    # Model config overrides
    parser.add_argument("--cls-dim", type=int, default=40,
                        help="Number of classes (40 for ModelNet40, 15 for ScanObjectNN)")
    parser.add_argument("--num-group", type=int, default=512,
                        help="Number of groups (512 for 8192 points, 64 for 2048 points)")
    parser.add_argument("--trans-dim", type=int, default=384,
                        help="Transformer dimension")
    parser.add_argument("--depth", type=int, default=12,
                        help="Transformer depth")
    parser.add_argument("--num-heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--group-size", type=int, default=32,
                        help="Group size for point grouping")
    args = parser.parse_args()

    import json
    import torch
    from easydict import EasyDict
    from models.Point_MAE import PointTransformer

    # Build config from arguments
    config = EasyDict({
        "trans_dim": args.trans_dim,
        "depth": args.depth,
        "drop_path_rate": 0.1,
        "cls_dim": args.cls_dim,
        "num_heads": args.num_heads,
        "group_size": args.group_size,
        "num_group": args.num_group,
        "encoder_dims": args.trans_dim,  # encoder_dims = trans_dim
    })

    if args.return_config:
        print(json.dumps(dict(config)))
        return

    # Build model
    model = PointTransformer(config)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract state dict
    if "base_model" in ckpt:
        state_dict = ckpt["base_model"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Check if this is a pre-trained MAE checkpoint (has MAE_encoder prefix)
    is_pretrain = any(k.startswith("MAE_encoder.") for k in state_dict.keys())

    if is_pretrain:
        print("Detected pre-trained MAE checkpoint, remapping keys...")
        # Remap MAE_encoder.* keys to classification model format
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("MAE_encoder."):
                # Remove MAE_encoder. prefix
                new_key = k.replace("MAE_encoder.", "")
                new_state_dict[new_key] = v
            # Skip decoder keys and mask_token (not used for classification)
            elif k.startswith("MAE_decoder.") or k.startswith("decoder_") or k == "mask_token":
                continue
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        # For pre-trained model, classifier head won't exist - load with strict=False
        # and randomly initialize the classifier
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained weights (missing {len(missing)} keys, unexpected {len(unexpected)} keys)")
        if missing:
            print(f"  Missing (will be random init): {missing[:5]}..." if len(missing) > 5 else f"  Missing: {missing}")
    else:
        # Fine-tuned checkpoint - load strictly
        model.load_state_dict(state_dict, strict=True)

    # Save full model object for unpickling in main process
    torch.save({
        "model": model,
        "state_dict": model.state_dict(),
        "config": dict(config),
        "model_class": "PointTransformer",
    }, args.output)

    print(f"Model saved to {args.output}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    main()
