#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-image-training
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         training/huggingface/hugging_face_distributor.py
# Description:  Uploads trained ResNet-34 model weights, Hugging Face Model Card
#               (README.md), and LICENSE to Hugging Face Hub.
# ==============================================================================

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Dynamic Path Resolution
SCRIPT_DIR = Path(__file__).resolve().parent           # /training/huggingface
TRAINING_DIR = SCRIPT_DIR.parent                      # /training
PROJECT_ROOT = TRAINING_DIR.parent                    # project root

DEFAULT_WEIGHTS_PATH = TRAINING_DIR / "trained_net_stream.pth"  # /training/trained_net_stream.pth (1.2 Flash)
README_PATH = SCRIPT_DIR / "README.md"                   # /training/huggingface/README.md
LICENSE_PATH = SCRIPT_DIR / "LICENSE"                    # /training/huggingface/LICENSE
ENV_PATH = PROJECT_ROOT / ".env"                         # .env

# Load environment variables if HF_TOKEN is defined in .env
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Target Hugging Face Repository
DEFAULT_REPO_ID = "RaulSalasSahuquillo/astroclass_ai"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish updated model weights, model card, and release tags to Hugging Face Hub."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.2-flash",
        help="Release version tag (e.g., v1.2-flash). Display name: AstroClass 1.2 Flash"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to trained .pth weights file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID (owner/repo)"
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip creating a git tag for this release"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    version = args.version
    repo_id = args.repo_id
    weights_path = args.weights

    print("  HUGGING FACE MODEL DISTRIBUTOR — ASTROCLASS AI\n")
    print(f"Target Repository:  https://huggingface.co/{repo_id}")
    print(f"Release Version:    {version}")
    print(f"Weights Source:     {weights_path}")
    print(f"Model Card Source:  {README_PATH}")
    print(f"License Source:     {LICENSE_PATH}")
    print("-" * 70)

    # Verify required files exist locally
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found at: {weights_path}")
        sys.exit(1)

    if not README_PATH.exists():
        print(f"[ERROR] Model Card README not found at: {README_PATH}")
        sys.exit(1)

    if not LICENSE_PATH.exists():
        print(f"[ERROR] LICENSE file not found at: {LICENSE_PATH}")
        sys.exit(1)

    # Initialize Hugging Face API client
    api = HfApi()

    # Create or verify repository on Hugging Face
    print("\n[1/5] Verifying/Creating Hugging Face model repository...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True
    )
    print("      Repository ready.")

    # Upload Model Weights (astroclass_ai.pth)
    print(f"\n[2/5] Uploading model weights as 'astroclass_ai.pth' ({weights_path.stat().st_size / (1024*1024):.2f} MB)...")
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="astroclass_ai.pth",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {version}: Update ResNet-34 model weights (astroclass_ai.pth)"
    )
    print("      Weights uploaded successfully.")

    # Upload Model Card (README.md)
    print("\n[3/5] Uploading Model Card as 'README.md'...")
    api.upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {version}: Update Hugging Face Model Card (README.md)"
    )
    print("      Model Card uploaded successfully.")

    # Upload License (LICENSE)
    print("\n[4/5] Uploading license file as 'LICENSE'...")
    api.upload_file(
        path_or_fileobj=str(LICENSE_PATH),
        path_in_repo="LICENSE",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {version}: Update CC BY-NC-SA 4.0 LICENSE file"
    )
    print("      LICENSE uploaded successfully.")

    # Create Git Release Tag on Hugging Face
    if not args.no_tag:
        print(f"\n[5/5] Creating Git release tag '{version}' on Hugging Face...")
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=version,
                tag_message=f"Release {version}: Updated ResNet-34 weights",
                exist_ok=True
            )
            print(f"      Tag '{version}' created successfully.")
        except Exception as e:
            print(f"      [WARNING] Could not create tag '{version}': {e}")
    else:
        print("\n[5/5] Skipping tag creation (--no-tag specified).")

    print("\n" + "=" * 70)
    print("  ALL ASSETS PUBLISHED SUCCESSFULLY!")
    print(f"  View your model on Hugging Face: https://huggingface.co/{repo_id}")
    print(f"  Release Tag: {version}")


if __name__ == "__main__":
    main()
