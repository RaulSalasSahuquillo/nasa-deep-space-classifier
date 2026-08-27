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

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# ------------------------------------------------------------------------------
# Dynamic Path Resolution
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # /training/huggingface
TRAINING_DIR = SCRIPT_DIR.parent                      # /training
PROJECT_ROOT = TRAINING_DIR.parent                    # project root

WEIGHTS_PATH = TRAINING_DIR / "trained_net.pth"       # /training/trained_net.pth
README_PATH = SCRIPT_DIR / "README.md"                # /training/huggingface/README.md
LICENSE_PATH = SCRIPT_DIR / "LICENSE"                 # /training/huggingface/LICENSE
ENV_PATH = PROJECT_ROOT / ".env"                      # .env

# Load environment variables if HF_TOKEN is defined in .env
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Target Hugging Face Repository
REPO_ID = "RaulSalasSahuquillo/astroclass_ai"


def main():
    print("=" * 70)
    print("  HUGGING FACE MODEL DISTRIBUTOR — ASTROCLASS AI")
    print("=" * 70)
    print(f"Target Repository:  https://huggingface.co/{REPO_ID}")
    print(f"Weights Source:     {WEIGHTS_PATH}")
    print(f"Model Card Source:  {README_PATH}")
    print(f"License Source:     {LICENSE_PATH}")
    print("-" * 70)

    # 1. Verify required files exist locally
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Model weights not found at: {WEIGHTS_PATH}")
        sys.exit(1)

    if not README_PATH.exists():
        print(f"[ERROR] Model Card README not found at: {README_PATH}")
        sys.exit(1)

    if not LICENSE_PATH.exists():
        print(f"[ERROR] LICENSE file not found at: {LICENSE_PATH}")
        sys.exit(1)

    # 2. Initialize Hugging Face API client
    api = HfApi()

    # 3. Create or verify repository on Hugging Face
    print("\n[1/4] Verifying/Creating Hugging Face model repository...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True
    )
    print("      Repository ready.")

    # 4. Upload Model Weights (astroclass_ai.pth)
    print(f"\n[2/4] Uploading model weights as 'astroclass_ai.pth' ({WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)...")
    api.upload_file(
        path_or_fileobj=str(WEIGHTS_PATH),
        path_in_repo="astroclass_ai.pth",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload ResNet-34 trained weights (astroclass_ai.pth)"
    )
    print("      Weights uploaded successfully.")

    # 5. Upload Model Card (README.md)
    print("\n[3/4] Uploading Model Card as 'README.md'...")
    api.upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update Hugging Face Model Card (README.md)"
    )
    print("      Model Card uploaded successfully.")

    # 6. Upload License (LICENSE)
    print("\n[4/4] Uploading license file as 'LICENSE'...")
    api.upload_file(
        path_or_fileobj=str(LICENSE_PATH),
        path_in_repo="LICENSE",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add CC BY-NC-SA 4.0 LICENSE file"
    )
    print("      LICENSE uploaded successfully.")

    print("\n" + "=" * 70)
    print("  ALL ASSETS PUBLISHED SUCCESSFULLY!")
    print(f"  View your model on Hugging Face: https://huggingface.co/{REPO_ID}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
