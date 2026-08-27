#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-image-training
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         src/validator/gemini_validator.py
# Description:  Automated validation script powered by Google Gemini API to audit
#               and verify neural network image classifications one by one.
# ==============================================================================

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress internal SDK advisory logs
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

# Google Gemini SDK imports (google-genai)
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_INSTALLED = True
except ImportError:
    GENAI_INSTALLED = False

# Locate project paths dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "network_sorting" / "classified_images.db"
DEFAULT_SORTING_DIR = PROJECT_ROOT / "network_sorting"

# Load environment variables from .env file at project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# Terminal ANSI Color Codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
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
    """Pydantic schema for structured Gemini API response."""
    is_correct: bool = Field(
        description="True if the neural network predicted class accurately matches what is shown in the image, False otherwise."
    )
    actual_class: str = Field(
        description="The true astronomical class observed in the image (star, galaxy, nebula, planet, quasar, or non-astronomical description)."
    )
    confidence: str = Field(
        description="Confidence level of the Gemini assessment: 'high', 'medium', or 'low'."
    )
    explanation: str = Field(
        description="Brief, concise explanation (1-2 sentences) in English describing what the image depicts and why the neural classification is correct or incorrect."
    )


def get_gemini_client(api_key: Optional[str] = None):
    """Initializes and returns the Google GenAI client."""
    if not GENAI_INSTALLED:
        print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] The 'google-genai' package is not installed.{Colors.RESET}")
        print("Please install it by running:")
        print(f"  {Colors.CYAN}pip install google-genai{Colors.RESET}\n")
        sys.exit(1)

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] No valid Gemini API key found.{Colors.RESET}")
        print(f"Please define {Colors.CYAN}GEMINI_API_KEY{Colors.RESET} in your {Colors.CYAN}.env{Colors.RESET} file or pass it via {Colors.CYAN}--api-key <YOUR_KEY>{Colors.RESET}.\n")
        print("To configure it safely in your terminal:")
        print(f'  printf "Enter GEMINI_API_KEY (typing hidden): " && read -s val && echo && echo "GEMINI_API_KEY=$val" >> "{PROJECT_ROOT}/.env" && echo "Saved."\n')
        sys.exit(1)

    return genai.Client(api_key=key)


def load_and_preprocess_image(image_path: Path, max_dimension: int = 1024) -> Image.Image:
    """Loads an image, converts it to RGB and resizes if larger than max_dimension to optimize API transfer."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    width, height = img.size
    if max(width, height) > max_dimension:
        ratio = max_dimension / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    return img


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
    classes = ['star', 'galaxy', 'nebula', 'planet', 'quasar']
    if category_filter:
        classes = [c for c in classes if c.lower() == category_filter.lower()]
    
    records = []
    item_id = 1
    for cat in classes:
        cat_dir = sorting_dir / cat
        if not cat_dir.exists():
            continue
        for file in sorted(os.listdir(cat_dir)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = cat_dir / file
                records.append({
                    'id': item_id,
                    'filename': file,
                    'source_directory': str(cat_dir),
                    'predicted_class': cat,
                    'destination_path': str(full_path),
                    'timestamp': datetime.fromtimestamp(full_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
                item_id += 1
    return records


def setup_validation_table(db_path: Path) -> None:
    """Ensures gemini_validations table exists in SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()


def save_validation_to_db(db_path: Path, record: Dict[str, Any], audit: ClassificationAudit, model_name: str) -> None:
    """Stores single validation result in SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO gemini_validations (
            classification_id, filename, predicted_class, is_correct, 
            gemini_class, confidence, explanation, model_used, validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.get('id'),
        record['filename'],
        record['predicted_class'],
        1 if audit.is_correct else 0,
        audit.actual_class,
        audit.confidence,
        audit.explanation,
        model_name,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def analyze_image_with_gemini(
    client: Any,
    image: Image.Image,
    filename: str,
    predicted_class: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> ClassificationAudit:
    """Sends image and classification task to Gemini API with retry logic."""
    prompt = f"""You are an expert astrophysicist specialized in observational astronomy and NASA/HST/JWST deep space imagery.
A convolutional neural network (ResNet-34) has classified the attached astronomical image ('{filename}').

Neural Network Predicted Class: '{predicted_class}'

The 5 astrophysical classes of this project are:
1. 'star': Single stars, stellar fields, open/globular clusters, star trails, meteors.
2. 'galaxy': Spiral, elliptical, irregular galaxies, galaxy clusters, interacting galaxies.
3. 'nebula': Emission, reflection, planetary nebulae, supernova remnants, interstellar gas & dust clouds.
4. 'planet': Solar system planets, moons, ring systems, planetary surface geology/craters, rover close-ups.
5. 'quasar': Active galactic nuclei (AGN), energetic pulsars, radio galaxies with relativistic jets.

Instructions:
1. Carefully inspect the attached deep space image.
2. Determine whether the neural network classification '{predicted_class}' is CORRECT (is_correct: true) or INCORRECT (is_correct: false).
3. If incorrect or non-astronomical (e.g. animal, artifact, diagram), indicate what it represents in 'actual_class'.
4. Provide a clear, concise rationale (1-2 sentences) in English in 'explanation'.
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClassificationAudit,
                    temperature=0.1,
                )
            )

            # Parse structured output
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed
            
            # Fallback if parsed isn't automatically populated
            raw_text = response.text.strip()
            data = json.loads(raw_text)
            return ClassificationAudit(**data)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                wait_time = attempt * 4
                print(f"    {Colors.YELLOW}[Quota/Rate Limit] Waiting {wait_time}s before retry...{Colors.RESET}")
                time.sleep(wait_time)
            elif attempt < max_retries:
                wait_time = attempt * 2
                print(f"    {Colors.YELLOW}[Notice] Retry {attempt}/{max_retries} following error: {err_str[:60]}... Waiting {wait_time}s{Colors.RESET}")
                time.sleep(wait_time)
            else:
                # Fallback audit if all retries fail
                return ClassificationAudit(
                    is_correct=False,
                    actual_class="evaluation_error",
                    confidence="low",
                    explanation=f"Error evaluating image with Gemini API after {max_retries} attempts: {err_str[:100]}"
                )


def run_validation(args: argparse.Namespace) -> None:
    """Main execution workflow for image validation."""
    no_color = args.no_color or not sys.stdout.isatty()

    # Initialize Gemini client
    client = get_gemini_client(args.api_key)

    # Retrieve records
    db_path = Path(args.db_path)
    records = []
    
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
            'id': 1,
            'filename': img_path.name,
            'source_directory': str(img_path.parent),
            'predicted_class': expected_class,
            'destination_path': str(img_path),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        setup_validation_table(db_path)

    # Banner
    print(f"{Colors.BOLD}{Colors.CYAN}\nNASA DEEP SPACE CLASSIFIER - GEMINI AI VALIDATION{Colors.RESET}")
    print(f"Gemini Model:        {Colors.BOLD}{args.model}{Colors.RESET}")
    print(f"Data Source:         {source_mode}")
    print(f"Total to Analyze:    {Colors.BOLD}{total_images}{Colors.RESET} images")
    if args.category:
        print(f"Category Filter:     {Colors.YELLOW}{args.category}{Colors.RESET}")
    print(f"Delay Between Calls: {args.delay:.1f}s")
    print("-" * 80)

    correct_count = 0
    incorrect_count = 0
    errors_count = 0
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
            img = load_and_preprocess_image(image_path)
            audit = analyze_image_with_gemini(
                client=client,
                image=img,
                filename=filename,
                predicted_class=predicted_class,
                model_name=args.model,
                max_retries=args.max_retries
            )

            status_str = format_status(audit.is_correct, no_color=no_color)
            
            if audit.is_correct:
                correct_count += 1
                category_stats[predicted_class]["correct"] += 1
            else:
                incorrect_count += 1
                category_stats[predicted_class]["incorrect"] += 1

            # Terminal Output for each image
            print(f"\n[{idx:03d}/{total_images:03d}] Image: {Colors.BOLD}{filename}{Colors.RESET}")
            print(f"          Neural Net:     {Colors.CYAN}{predicted_class:<10}{Colors.RESET} ➔ Is correct?: {status_str}")
            print(f"          Gemini Detected:{Colors.YELLOW}{audit.actual_class}{Colors.RESET} (Confidence: {audit.confidence})")
            print(f"          Explanation:    {Colors.DIM}{audit.explanation}{Colors.RESET}")

            # Record result
            res_item = {
                "id": record.get("id", idx),
                "filename": filename,
                "predicted_class": predicted_class,
                "is_correct": audit.is_correct,
                "gemini_class": audit.actual_class,
                "confidence": audit.confidence,
                "explanation": audit.explanation,
                "image_path": str(image_path)
            }
            results_report.append(res_item)

            if args.update_db and db_path.exists():
                save_validation_to_db(db_path, record, audit, args.model)

        except Exception as ex:
            print(f"\n[{idx:03d}/{total_images:03d}] Image: {Colors.BOLD}{filename}{Colors.RESET}")
            print(f"          {Colors.RED}✗ Error processing image: {ex}{Colors.RESET}")
            errors_count += 1

        # Delay to stay within quota
        if idx < total_images and args.delay > 0:
            time.sleep(args.delay)

    elapsed_time = time.time() - start_time
    analyzed_total = correct_count + incorrect_count
    accuracy = (correct_count / analyzed_total * 100) if analyzed_total > 0 else 0.0

    # Summary Section
    print(f"{Colors.BOLD}{Colors.CYAN}\nFINAL VALIDATION SUMMARY{Colors.RESET}")
    print(f"Images Analyzed:        {Colors.BOLD}{analyzed_total}{Colors.RESET} / {total_images}")
    print(f"CORRECT Classifications:   {Colors.GREEN}{Colors.BOLD}{correct_count}{Colors.RESET} ({accuracy:.1f}%)")
    print(f"INCORRECT Classifications: {Colors.RED}{Colors.BOLD}{incorrect_count}{Colors.RESET} ({100.0 - accuracy:.1f}%)")
    if errors_count > 0:
        print(f"Read/API Errors:        {Colors.YELLOW}{errors_count}{Colors.RESET}")
    print(f"Elapsed Time:           {elapsed_time:.1f} seconds ({elapsed_time / max(1, analyzed_total):.2f}s/img)")
    print("-" * 80)

    # Category breakdown table
    print(f"{Colors.BOLD}Breakdown by Category:{Colors.RESET}")
    print(f"{'Category':<12} | {'Total':<6} | {'Correct':<10} | {'Incorrect':<12} | {'Accuracy':<10}")
    print("-" * 60)
    for cat, stats in sorted(category_stats.items()):
        c_tot = stats["total"]
        c_cor = stats["correct"]
        c_inc = stats["incorrect"]
        c_acc = (c_cor / c_tot * 100) if c_tot > 0 else 0.0
        print(f"{cat:<12} | {c_tot:<6} | {c_cor:<10} | {c_inc:<12} | {c_acc:>7.1f}%")
    print("=" * 80)

    # Highlight misclassified files
    misclassified = [r for r in results_report if not r["is_correct"]]
    if misclassified:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Misclassified Images ({len(misclassified)} found):{Colors.RESET}")
        for item in misclassified[:15]:
            print(f"  • {item['filename']} -> Neural Net: [{item['predicted_class']}], Gemini: [{item['gemini_class']}]")
            print(f"    └─ {Colors.DIM}{item['explanation']}{Colors.RESET}")
        if len(misclassified) > 15:
            print(f"  ... and {len(misclassified) - 15} more.")

    # Save JSON report if requested
    if args.save_report:
        report_path = Path(args.save_report)
        report_data = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "model": args.model,
                "total_analyzed": analyzed_total,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "accuracy_percent": accuracy,
                "elapsed_seconds": elapsed_time
            },
            "category_stats": category_stats,
            "results": results_report
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n{Colors.GREEN}✓ Report successfully saved to: {report_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Audits and validates deep space image classifications from neural network using Google Gemini API."
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
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)."
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Delay in seconds between API requests to respect quota (default: 1.0s)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})."
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
        help="Path to a single image file for individual verification."
    )
    parser.add_argument(
        "--expected-class",
        type=str,
        default=None,
        help="Expected category when analyzing a single image with --image."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API Key (optional, defaults to GEMINI_API_KEY in .env)."
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
        help="Saves validation audit records to 'gemini_validations' table in SQLite DB."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per image for transient or rate-limit errors (default: 3)."
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
