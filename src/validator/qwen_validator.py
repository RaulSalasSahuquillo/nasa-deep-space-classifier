#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-image-training
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         src/validator/qwen_validator.py
# Description:  Standalone in-process Qwen AI validator and online neural network
#               trainer. Evaluates classified images image-by-image and directly
#               adjusts ResNet-34 weights based on astrophysical assessments.
#               Runs 100% locally with zero external software or daemons.
# ==============================================================================

import argparse
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet34

from dotenv import load_dotenv

# Suppress verbose transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "network_sorting" / "classified_images.db"
DEFAULT_SORTING_DIR = PROJECT_ROOT / "network_sorting"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "training" / "trained_net.pth"

# Load environment variables from .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# 5 Project Astronomical Classes (Index mapping matches training/network_training.ipynb)
CLASS_NAMES = ["star", "galaxy", "quasar", "nebula", "planet"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Default Qwen Model Cache & Fallback Identifiers
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def get_default_qwen_model_path() -> str:
    """
    Dynamically resolves the default Qwen model path or HuggingFace repo identifier.
    1. Checks the QWEN_MODEL_PATH environment variable if defined in .env.
    2. Checks the user's local HuggingFace cache dynamically via Path.home().
    3. Falls back to the official HuggingFace repository ID.
    Avoids hardcoding machine-specific user home directories into the codebase.
    """
    env_model = os.getenv("QWEN_MODEL_PATH")
    if env_model:
        return env_model

    local_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
        / "snapshots"
        / "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
    if local_snapshot.exists():
        return str(local_snapshot)

    return DEFAULT_QWEN_MODEL_ID



# Terminal ANSI Color Codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def format_status(is_correct: bool, no_color: bool = False) -> str:
    """Returns formatted string indicating whether classification is correct."""
    if no_color:
        return "CORRECT" if is_correct else "INCORRECT"
    if is_correct:
        return f"{Colors.GREEN}{Colors.BOLD}✓ CORRECT{Colors.RESET}"
    return f"{Colors.RED}{Colors.BOLD}✗ INCORRECT{Colors.RESET}"


class ClassificationAudit(BaseModel):
    """Pydantic schema for structured Qwen evaluation response."""
    is_correct: bool = Field(
        description="True if the neural network predicted class accurately matches what is shown in the image, False otherwise."
    )
    actual_class: str = Field(
        description="The true astronomical class observed in the image (star, galaxy, nebula, planet, quasar, or non-astronomical description)."
    )
    confidence: str = Field(
        description="Confidence level of the Qwen assessment: 'high', 'medium', or 'low'."
    )
    explanation: str = Field(
        description="Brief, concise explanation (1-2 sentences) in English describing what the image depicts and why the neural classification is correct or incorrect."
    )


# Astrophysical Feature Extractor
def extract_astrophysical_features(image: Image.Image) -> Dict[str, Any]:
    """
    Extracts photometric, spatial, and morphological descriptors from an astronomical
    image to provide rich perceptual context to Qwen.
    """
    rgb_img = image.convert("RGB")
    width, height = rgb_img.size
    arr = np.array(rgb_img, dtype=np.float32)

    # Photometric statistics
    luminance = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    mean_lum = float(np.mean(luminance))
    std_lum = float(np.std(luminance))
    max_lum = float(np.max(luminance))
    dark_ratio = float(np.mean(luminance < 20.0))  # Fraction of space background

    # Color channel dynamics
    mean_r = float(np.mean(arr[:, :, 0]))
    mean_g = float(np.mean(arr[:, :, 1]))
    mean_b = float(np.mean(arr[:, :, 2]))
    rb_ratio = float((mean_r + 1e-4) / (mean_b + 1e-4))

    # Spatial concentration (core vs periphery)
    h_mid, w_mid = height // 2, width // 2
    dh, dw = max(1, height // 6), max(1, width // 6)
    core = luminance[h_mid - dh : h_mid + dh, w_mid - dw : w_mid + dw]
    core_mean = float(np.mean(core)) if core.size > 0 else mean_lum
    concentration_ratio = float(core_mean / (mean_lum + 1e-4))

    # High-intensity point count (potential stars / point sources)
    bright_threshold = mean_lum + 3.0 * std_lum
    num_bright_peaks = int(np.sum(luminance > max(bright_threshold, 180.0)))

    # Simple texture gradient (Laplacian proxy for diffuse vs sharp)
    diff_y = np.abs(np.diff(luminance, axis=0))
    diff_x = np.abs(np.diff(luminance, axis=1))
    edge_density = float((np.mean(diff_y) + np.mean(diff_x)) / 2.0)

    return {
        "dimensions": f"{width}x{height}",
        "mean_luminance": round(mean_lum, 1),
        "luminance_std": round(std_lum, 1),
        "dark_space_background_pct": round(dark_ratio * 100, 1),
        "rgb_means": f"R:{mean_r:.1f} G:{mean_g:.1f} B:{mean_b:.1f}",
        "red_blue_ratio": round(rb_ratio, 2),
        "core_concentration_index": round(concentration_ratio, 2),
        "prominent_bright_pixels": num_bright_peaks,
        "edge_contrast_metric": round(edge_density, 2),
    }


# In-Process Qwen Model Engine
class InProcessQwenEngine:
    """Manages in-process Qwen model loading and inference via Hugging Face transformers."""
    def __init__(self, model_path: Optional[Any] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else Path(get_default_qwen_model_path())
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        target_path = str(self.model_path)
        display_path = target_path
        home_str = str(Path.home())
        if display_path.startswith(home_str):
            display_path = "~" + display_path[len(home_str):]
        print(f"{Colors.CYAN}Loading Qwen AI into memory ({display_path})...{Colors.RESET}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            target_path,
            local_files_only=os.path.exists(target_path)
        )

        if self.device == "cuda":
            # 4-bit quantization via bitsandbytes to run within ~4.5GB VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                target_path,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=os.path.exists(target_path)
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                local_files_only=os.path.exists(target_path)
            )
        self.model.eval()
        print(f"{Colors.GREEN}✓ Qwen AI successfully loaded into GPU memory.{Colors.RESET}")

    def evaluate_classification(
        self,
        filename: str,
        predicted_class: str,
        features: Dict[str, Any]
    ) -> ClassificationAudit:
        """Sends classification evaluation prompt to in-process Qwen and parses structured response."""
        prompt = f"""You are an expert astrophysicist and deep learning auditor specialized in NASA, HST, and JWST space imagery.
A convolutional neural network (ResNet-34) classified the deep space image '{filename}' as: '{predicted_class}'.

Image Physical & Photometric Features:
- Dimensions: {features['dimensions']}
- Mean Luminance: {features['mean_luminance']} (Std: {features['luminance_std']})
- Dark Space Background: {features['dark_space_background_pct']}%
- RGB Channel Balance: {features['rgb_means']} (Red/Blue Ratio: {features['red_blue_ratio']})
- Central Concentration Index: {features['core_concentration_index']}
- Edge/Texture Metric: {features['edge_contrast_metric']}

The 5 target classes for this NASA project are:
1. 'star': Single stars with diffraction spikes, star trails, stellar clusters, stellar fields.
2. 'galaxy': Spiral arms, elliptical disks, central galactic bulge, galaxy clusters, interacting mergers.
3. 'nebula': Extended diffuse gas, glowing emission/reflection clouds, dark absorption dust, planetary nebulae, supernova remnants.
4. 'planet': Smooth or cratered round disks, rings, solar system planetary bodies, moons, planetary surface geology.
5. 'quasar': Energetic active galactic nuclei (AGN), core relativistic jets, luminous point-like galactic cores.

Evaluate whether the neural network prediction '{predicted_class}' is accurate based on these features and filename context.
Respond ONLY with a valid JSON object strictly matching this schema:
{{
  "is_correct": true or false,
  "actual_class": "star" or "galaxy" or "nebula" or "planet" or "quasar",
  "confidence": "high" or "medium" or "low",
  "explanation": "1-2 concise sentences in English explaining the astronomical reasoning."
}}"""

        messages = [
            {"role": "system", "content": "You are an expert astrophysicist. Output only raw JSON."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        raw_output = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return self._parse_qwen_response(raw_output, predicted_class)

    def _parse_qwen_response(self, raw_text: str, fallback_predicted: str) -> ClassificationAudit:
        """Parses Qwen's generated text into a validated ClassificationAudit."""
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                # Validate actual_class string
                actual = str(data.get("actual_class", fallback_predicted)).lower().strip()
                # Find matching target class if embedded in text
                for c in CLASS_NAMES:
                    if c in actual:
                        actual = c
                        break
                if actual not in CLASS_NAMES:
                    actual = fallback_predicted

                is_correct = bool(data.get("is_correct", actual == fallback_predicted.lower()))
                conf = str(data.get("confidence", "medium")).lower().strip()
                if conf not in ["high", "medium", "low"]:
                    conf = "medium"
                explanation = str(data.get("explanation", "Evaluated by local Qwen AI.")).strip()

                return ClassificationAudit(
                    is_correct=is_correct,
                    actual_class=actual,
                    confidence=conf,
                    explanation=explanation
                )
            except Exception:
                pass

        # Text fallback parsing
        is_corr = "correct" in raw_text.lower() and "incorrect" not in raw_text.lower()
        detected_class = fallback_predicted
        for c in CLASS_NAMES:
            if re.search(r"\b" + c + r"\b", raw_text.lower()):
                detected_class = c
                break

        return ClassificationAudit(
            is_correct=is_corr,
            actual_class=detected_class,
            confidence="medium",
            explanation=raw_text.replace("\n", " ")[:150]
        )


# ResNet-34 Neural Network Trainer
class ResNetTrainer:
    """Loads ResNet-34 and performs active online weight adjustments based on Qwen supervision."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS_PATH,
        learning_rate: float = 0.0001,
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = Path(weights_path)
        self.lr = learning_rate

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.net = resnet34(weights=None)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

        self._load_weights()
        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def _load_weights(self) -> None:
        if self.weights_path.exists():
            # Create safe backup if not already present
            backup_path = self.weights_path.with_suffix(".pth.backup")
            if not backup_path.exists():
                shutil.copy2(self.weights_path, backup_path)
                print(f"{Colors.DIM}Backed up initial weights to: {backup_path}{Colors.RESET}")

            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"{Colors.GREEN}✓ ResNet-34 weights loaded from: {self.weights_path}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}[!] Weights file not found at {self.weights_path}. Initializing with base architecture.{Colors.RESET}")

    def evaluate_image(self, img: Image.Image) -> Tuple[str, float, torch.Tensor, torch.Tensor]:
        """Runs forward pass on image. Returns (predicted_class, confidence, probabilities, logits)."""
        self.net.eval()
        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.net(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        return CLASS_NAMES[pred_idx], probs[pred_idx].item(), probs, logits

    def adjust_weights_with_qwen(
        self,
        img: Image.Image,
        target_class: str,
        confidence_level: str
    ) -> Dict[str, float]:
        """
        Executes a backpropagation and optimizer step on ResNet-34 using Qwen's target label.
        Returns loss before and after, as well as target probability improvement.
        """
        if target_class not in CLASS_TO_IDX:
            return {"loss_before": 0.0, "loss_after": 0.0, "prob_before": 0.0, "prob_after": 0.0}

        target_idx = CLASS_TO_IDX[target_class]
        target_tensor = torch.tensor([target_idx], device=self.device)

        # Scale update magnitude by Qwen's confidence
        conf_weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
        conf_scale = conf_weights.get(confidence_level.lower(), 0.7)

        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        # Pre-update state
        self.net.eval()
        with torch.no_grad():
            initial_logits = self.net(img_tensor)
            initial_loss = self.criterion(initial_logits, target_tensor).item()
            initial_probs = torch.softmax(initial_logits, dim=1)[0]
            prob_before = initial_probs[target_idx].item()

        # Training Step (Qwen Active Weight Update)
        self.net.train()
        # Freeze BatchNorm running statistics during single-sample online training
        for m in self.net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

        self.optimizer.zero_grad()
        logits = self.net(img_tensor)
        loss = self.criterion(logits, target_tensor) * conf_scale
        loss.backward()
        self.optimizer.step()

        # Post-update state
        self.net.eval()
        with torch.no_grad():
            new_logits = self.net(img_tensor)
            new_loss = self.criterion(new_logits, target_tensor).item()
            new_probs = torch.softmax(new_logits, dim=1)[0]
            prob_after = new_probs[target_idx].item()

        return {
            "loss_before": initial_loss,
            "loss_after": new_loss,
            "prob_before": prob_before,
            "prob_after": prob_after,
            "prob_delta": prob_after - prob_before,
            "loss_reduction": initial_loss - new_loss
        }

    def save_weights(self, output_path: Optional[Path] = None) -> Path:
        """Saves updated model weights to disk."""
        target_path = Path(output_path or self.weights_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), target_path)
        return target_path


# Data Loading & Database Storage
def get_classified_images_from_db(db_path: Path, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves classification records from SQLite database."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'")
    if not cursor.fetchone():
        conn.close()
        return []

    if category_filter:
        cursor.execute(
            "SELECT id, filename, source_directory, predicted_class, destination_path, timestamp FROM classifications WHERE predicted_class = ? ORDER BY id ASC",
            (category_filter,)
        )
    else:
        cursor.execute(
            "SELECT id, filename, source_directory, predicted_class, destination_path, timestamp FROM classifications ORDER BY id ASC"
        )

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_classified_images_from_folders(sorting_dir: Path, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fallback: scans category folders directly under network_sorting/."""
    classes = CLASS_NAMES[:]
    if category_filter:
        classes = [c for c in classes if c.lower() == category_filter.lower()]

    records = []
    item_id = 1
    for cat in classes:
        cat_dir = sorting_dir / cat
        if not cat_dir.exists():
            continue
        for file in sorted(os.listdir(cat_dir)):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = cat_dir / file
                records.append({
                    "id": item_id,
                    "filename": file,
                    "source_directory": str(cat_dir),
                    "predicted_class": cat,
                    "destination_path": str(full_path),
                    "timestamp": datetime.fromtimestamp(full_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
                item_id += 1
    return records


def setup_validation_tables(db_path: Path) -> None:
    """Ensures qwen_validations and legacy gemini_validations tables exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Primary Qwen validations table with training metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qwen_validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER,
            filename TEXT,
            predicted_class TEXT,
            is_correct INTEGER,
            qwen_class TEXT,
            confidence TEXT,
            explanation TEXT,
            loss_before REAL,
            loss_after REAL,
            prob_before REAL,
            prob_after REAL,
            weights_updated INTEGER,
            model_used TEXT,
            validated_at TEXT
        )
    """)

    # Backwards-compatible legacy table for dashboard viewer
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gemini_validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER,
            filename TEXT,
            predicted_class TEXT,
            is_correct INTEGER,
            gemini_class TEXT,
            confidence TEXT,
            explanation TEXT,
            model_used TEXT,
            validated_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_validation_to_db(
    db_path: Path,
    record: Dict[str, Any],
    audit: ClassificationAudit,
    training_metrics: Optional[Dict[str, float]],
    model_name: str
) -> None:
    """Stores audit and weight update records into SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    t_met = training_metrics or {}
    loss_b = t_met.get("loss_before", 0.0)
    loss_a = t_met.get("loss_after", 0.0)
    prob_b = t_met.get("prob_before", 0.0)
    prob_a = t_met.get("prob_after", 0.0)
    updated = 1 if t_met else 0

    cursor.execute("""
        INSERT INTO qwen_validations (
            classification_id, filename, predicted_class, is_correct,
            qwen_class, confidence, explanation,
            loss_before, loss_after, prob_before, prob_after,
            weights_updated, model_used, validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("id"),
        record["filename"],
        record["predicted_class"],
        1 if audit.is_correct else 0,
        audit.actual_class,
        audit.confidence,
        audit.explanation,
        loss_b,
        loss_a,
        prob_b,
        prob_a,
        updated,
        model_name,
        now_str
    ))

    # Also log to legacy table for backwards compatibility
    cursor.execute("""
        INSERT INTO gemini_validations (
            classification_id, filename, predicted_class, is_correct,
            gemini_class, confidence, explanation, model_used, validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("id"),
        record["filename"],
        record["predicted_class"],
        1 if audit.is_correct else 0,
        audit.actual_class,
        audit.confidence,
        audit.explanation,
        f"qwen-{model_name}",
        now_str
    ))

    conn.commit()
    conn.close()


# Main Validation & Training Loop
def run_validation(args: argparse.Namespace) -> None:
    """Main workflow: audits images, invokes in-process Qwen, and adjusts ResNet-34 weights."""
    no_color = args.no_color or not sys.stdout.isatty()

    # Determine Model Path
    model_path = Path(args.model_path or get_default_qwen_model_path())

    # Initialize in-process Qwen Engine
    qwen = InProcessQwenEngine(model_path=model_path)

    # Initialize ResNet-34 Trainer
    trainer = ResNetTrainer(
        weights_path=Path(args.weights_path),
        learning_rate=args.lr
    )

    # Retrieve records
    db_path = Path(args.db_path)
    records = []
    source_mode = ""

    if db_path.exists() and not args.force_folder:
        records = get_classified_images_from_db(db_path, args.category)
        source_mode = f"SQLite Database ({db_path})"

    if not records:
        records = get_classified_images_from_folders(DEFAULT_SORTING_DIR, args.category)
        source_mode = f"'network_sorting/' folders ({DEFAULT_SORTING_DIR})"

    # Handle single image verification mode if specified
    if args.image:
        img_path = Path(args.image).resolve()
        if not img_path.exists():
            print(f"{Colors.RED}[ERROR] Image file does not exist: {img_path}{Colors.RESET}")
            sys.exit(1)
        expected_class = args.expected_class or img_path.parent.name
        records = [{
            "id": 1,
            "filename": img_path.name,
            "source_directory": str(img_path.parent),
            "predicted_class": expected_class,
            "destination_path": str(img_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]
        source_mode = f"Single Image: {img_path.name}"

    if not records:
        print(f"{Colors.YELLOW}[!] No classified images found to analyze.{Colors.RESET}")
        print("Please run classification first using 'network_sorting/image_classifier.ipynb'.")
        sys.exit(0)

    # Apply limit if set
    if args.limit and args.limit > 0:
        records = records[:args.limit]

    total_images = len(records)

    # Optional DB logging setup
    if args.update_db and db_path.exists():
        setup_validation_tables(db_path)

    # Banner
    model_display = model_path.name if model_path.exists() else str(args.model)
    print(f"{Colors.BOLD}{Colors.CYAN}\nNASA DEEP SPACE CLASSIFIER - QWEN AI VALIDATOR & NEURAL TRAINER{Colors.RESET}")
    print(f"Qwen Model:          {Colors.BOLD}{model_display}{Colors.RESET} (In-Process, 4-bit CUDA)")
    print(f"Neural Network:      ResNet-34 ({trainer.weights_path.name})")
    print(f"Active Weight Edit:  {Colors.GREEN if not args.no_weight_updates else Colors.YELLOW}{'ENABLED' if not args.no_weight_updates else 'DISABLED (Dry Run)'}{Colors.RESET} (lr={args.lr})")
    print(f"Data Source:         {source_mode}")
    print(f"Total to Analyze:    {Colors.BOLD}{total_images}{Colors.RESET} images")
    if args.category:
        print(f"Category Filter:     {Colors.YELLOW}{args.category}{Colors.RESET}")
    print("-" * 85)

    correct_count = 0
    incorrect_count = 0
    errors_count = 0
    weights_adjusted_count = 0
    total_loss_reduction = 0.0
    results_report = []

    category_stats = {}
    start_time = time.time()

    for idx, record in enumerate(records, start=1):
        filename = record["filename"]
        predicted_class = record["predicted_class"]
        image_path = Path(record["destination_path"])

        if cat := predicted_class:
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0, "incorrect": 0}
            category_stats[cat]["total"] += 1

        # Check if file exists physically
        if not image_path.exists():
            print(f"[{idx:03d}/{total_images:03d}] {Colors.BOLD}{filename}{Colors.RESET}")
            print(f"          {Colors.RED}✗ File not found on disk: {image_path}{Colors.RESET}")
            errors_count += 1
            continue

        try:
            pil_img = Image.open(image_path)

            # Pre-update ResNet inference
            net_pred, net_conf, _, _ = trainer.evaluate_image(pil_img)

            # Extract astrophysical features & ask Qwen
            features = extract_astrophysical_features(pil_img)
            audit = qwen.evaluate_classification(
                filename=filename,
                predicted_class=predicted_class,
                features=features
            )

            status_str = format_status(audit.is_correct, no_color=no_color)

            if audit.is_correct:
                correct_count += 1
                category_stats[predicted_class]["correct"] += 1
            else:
                incorrect_count += 1
                category_stats[predicted_class]["incorrect"] += 1

            # Step 3: Active weight adjustment if enabled
            training_metrics = None
            if not args.no_weight_updates:
                training_metrics = trainer.adjust_weights_with_qwen(
                    img=pil_img,
                    target_class=audit.actual_class,
                    confidence_level=audit.confidence
                )
                weights_adjusted_count += 1
                total_loss_reduction += training_metrics.get("loss_reduction", 0.0)

            # Terminal Output for each image
            print(f"\n[{idx:03d}/{total_images:03d}] Image: {Colors.BOLD}{filename}{Colors.RESET}")
            print(f"          Neural Net:     {Colors.CYAN}{predicted_class:<10}{Colors.RESET} (P={net_conf*100:.1f}%) ➔ {status_str}")
            print(f"          Qwen Detected:  {Colors.YELLOW}{audit.actual_class:<10}{Colors.RESET} (Confidence: {audit.confidence})")
            print(f"          Explanation:    {Colors.DIM}{audit.explanation}{Colors.RESET}")

            if training_metrics:
                l_before = training_metrics["loss_before"]
                l_after = training_metrics["loss_after"]
                p_before = training_metrics["prob_before"] * 100
                p_after = training_metrics["prob_after"] * 100
                delta = training_metrics["prob_delta"] * 100
                sign = "+" if delta >= 0 else ""
                print(f"          {Colors.MAGENTA}⚡ Qwen Weight Step:{Colors.RESET} Loss {l_before:.3f} ➔ {l_after:.3f} | Target [{audit.actual_class}] Prob: {p_before:.1f}% ➔ {p_after:.1f}% ({sign}{delta:.1f}%)")

            # Record result
            res_item = {
                "id": record.get("id", idx),
                "filename": filename,
                "predicted_class": predicted_class,
                "is_correct": audit.is_correct,
                "qwen_class": audit.actual_class,
                "confidence": audit.confidence,
                "explanation": audit.explanation,
                "training_metrics": training_metrics,
                "image_path": str(image_path)
            }
            results_report.append(res_item)

            if args.update_db and db_path.exists():
                save_validation_to_db(db_path, record, audit, training_metrics, model_display)

        except Exception as ex:
            print(f"\n[{idx:03d}/{total_images:03d}] Image: {Colors.BOLD}{filename}{Colors.RESET}")
            print(f"          {Colors.RED}✗ Error processing image: {ex}{Colors.RESET}")
            errors_count += 1

        # Optional delay
        if idx < total_images and args.delay > 0:
            time.sleep(args.delay)

    # Save final model weights
    if not args.no_weight_updates and weights_adjusted_count > 0:
        saved_file = trainer.save_weights()
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Neural network weights successfully updated and saved to: {saved_file}{Colors.RESET}")

    elapsed_time = time.time() - start_time
    analyzed_total = correct_count + incorrect_count
    accuracy = (correct_count / analyzed_total * 100) if analyzed_total > 0 else 0.0

    # Summary Section
    print(f"{Colors.BOLD}{Colors.CYAN}\nFINAL VALIDATION & NEURAL TRAINING SUMMARY{Colors.RESET}")
    print(f"Images Analyzed:           {Colors.BOLD}{analyzed_total}{Colors.RESET} / {total_images}")
    print(f"CORRECT Classifications:      {Colors.GREEN}{Colors.BOLD}{correct_count}{Colors.RESET} ({accuracy:.1f}%)")
    print(f"INCORRECT Classifications:    {Colors.RED}{Colors.BOLD}{incorrect_count}{Colors.RESET} ({100.0 - accuracy:.1f}%)")
    if not args.no_weight_updates:
        print(f"Neural Weights Adjusted:   {Colors.MAGENTA}{Colors.BOLD}{weights_adjusted_count}{Colors.RESET} gradient updates applied")
        avg_loss_red = total_loss_reduction / max(1, weights_adjusted_count)
        print(f"Average Loss Reduction:    {Colors.GREEN}{avg_loss_red:.4f}{Colors.RESET} per step")
    if errors_count > 0:
        print(f"Read/Processing Errors:    {Colors.YELLOW}{errors_count}{Colors.RESET}")
    print(f"Elapsed Time:              {elapsed_time:.1f} seconds ({elapsed_time / max(1, analyzed_total):.2f}s/img)")
    print("-" * 85)

    # Category breakdown table
    print(f"{Colors.BOLD}Breakdown by Category:{Colors.RESET}")
    print(f"{'Category':<12} | {'Total':<6} | {'Correct':<10} | {'Incorrect':<12} | {'Accuracy':<10}")
    print("-" * 65)
    for cat, stats in sorted(category_stats.items()):
        c_tot = stats["total"]
        c_cor = stats["correct"]
        c_inc = stats["incorrect"]
        c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
        print(f"{cat:<12} | {c_tot:<6} | {c_cor:<10} | {c_inc:<12} | {c_acc:>7.1f}%")
    print("=" * 85)

    # Highlight misclassified files
    misclassified = [r for r in results_report if not r["is_correct"]]
    if misclassified:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Misclassified Images Evaluated by Qwen ({len(misclassified)} found):{Colors.RESET}")
        for item in misclassified[:15]:
            print(f"  • {item['filename']} -> Neural Net: [{item['predicted_class']}], Qwen: [{item['qwen_class']}]")
            print(f"    └─ {Colors.DIM}{item['explanation']}{Colors.RESET}")
        if len(misclassified) > 15:
            print(f"  ... and {len(misclassified) - 15} more.")

    # Save JSON report if requested
    if args.save_report:
        report_path = Path(args.save_report)
        try:
            rel_weights_path = str(trainer.weights_path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            rel_weights_path = trainer.weights_path.name

        cleaned_results = []
        for r in results_report:
            item = dict(r)
            if "image_path" in item:
                try:
                    item["image_path"] = str(Path(item["image_path"]).resolve().relative_to(PROJECT_ROOT.resolve()))
                except ValueError:
                    item["image_path"] = Path(item["image_path"]).name
            cleaned_results.append(item)

        report_data = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "validator_model": model_display,
                "weights_path": rel_weights_path,
                "total_analyzed": analyzed_total,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "weights_adjusted_count": weights_adjusted_count,
                "accuracy_percent": accuracy,
                "elapsed_seconds": elapsed_time
            },
            "category_stats": category_stats,
            "results": cleaned_results
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n{Colors.GREEN}✓ Report successfully saved to: {report_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="In-Process Qwen AI Validator & ResNet-34 Neural Network Active Trainer."
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of images to process (e.g. --limit 10). Defaults to all."
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        choices=["star", "galaxy", "nebula", "planet", "quasar"],
        help="Filter by specific astronomical category."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen-local",
        help="Qwen model name or identifier (default: qwen-local)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local Qwen model weights snapshot or HuggingFace repo ID (defaults to QWEN_MODEL_PATH or local cache)."
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to ResNet-34 weights .pth file (default: training/trained_net.pth)."
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate for Qwen-driven weight adjustment steps (default: 0.0001)."
    )
    parser.add_argument(
        "--no-weight-updates",
        action="store_true",
        help="Disables neural network weight modifications (audit-only mode)."
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.0,
        help="Delay in seconds between image processing (default: 0.0s for local AI)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database (default: network_sorting/classified_images.db)."
    )
    parser.add_argument(
        "--force-folder",
        action="store_true",
        help="Bypasses SQLite database and scans network_sorting/<class> folders directly."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to a single image file for individual verification and training."
    )
    parser.add_argument(
        "--expected-class",
        type=str,
        default=None,
        help="Expected category when analyzing a single image with --image."
    )
    parser.add_argument(
        "--save-report", "-o",
        type=str,
        default=None,
        help="Path to JSON file where validation report will be saved."
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Saves validation audit and weight update records to SQLite DB."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Ignored (retained for backward compatibility; Qwen runs locally without API keys)."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retained for backward compatibility."
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disables ANSI colors in terminal output."
    )

    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
