#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-deep-space-classifier
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         src/data_collection/image_collection.py
# Description:  Automated astronomical data harvester for NASA APOD and MAST
#               science observation archives with registry deduplication.
# ==============================================================================

import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from astropy.time import Time
from astroquery.mast import Observations

# Locate project root dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Load environment variables from .env file at project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Configuration
BASE_IMAGES_FOLDER = PROJECT_ROOT / "images"
NASA_FOLDER = BASE_IMAGES_FOLDER / "nasaDownloads"
MAST_FOLDER = BASE_IMAGES_FOLDER / "mastDownloads"

NASA_REGISTRY_FILE = NASA_FOLDER / "downloaded.json"
MAST_REGISTRY_FILE = MAST_FOLDER / "downloaded.json"

TARGET_PER_SOURCE = 100

APOD_API_URL = "https://api.nasa.gov/planetary/apod"
APOD_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

VALID_KEYWORDS = [
    "galaxy", "nebula", "planet", "jupiter", "saturn", "mars",
    "cluster", "supernova", "messier", "ngc", "star", "hubble", "webb"
]


def load_registry(filepath: Path) -> set:
    if filepath.exists():
        with open(filepath, "r") as f:
            return set(json.load(f))
    return set()


def save_registry(registry: set, filepath: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(sorted(registry), f, indent=2)


def is_relevant_object(name: str) -> bool:
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in VALID_KEYWORDS)


# NASA APOD API -> 100 Images
def fetch_apod_images(target_count: int) -> int:
    print(f"Fetching {target_count} High-Res Images from NASA APOD")
    registry = load_registry(NASA_REGISTRY_FILE)
    NASA_FOLDER.mkdir(parents=True, exist_ok=True)
    
    downloaded_in_session = 0

    while len(registry) < target_count:
        needed = target_count - len(registry)
        print(f"  [APOD Progress: {len(registry)}/{target_count}] Requesting batch from NASA...")
        
        params = {"api_key": APOD_API_KEY, "count": min(needed * 2, 100)}

        try:
            response = requests.get(APOD_API_URL, params=params, timeout=20)
            if response.status_code != 200:
                print(f"  Error accessing APOD API: HTTP {response.status_code}")
                break

            data = response.json()
            new_added_in_batch = 0

            for item in data:
                if len(registry) >= target_count:
                    break

                if item.get("media_type") != "image":
                    continue

                title = item.get("title", "")
                url = item.get("hdurl") or item.get("url")
                if not url:
                    continue

                filename = url.split("/")[-1]
                if filename in registry or not is_relevant_object(title):
                    continue

                print(f"  Downloading: '{title}'...")
                img_data = requests.get(url, timeout=30).content
                filepath = NASA_FOLDER / filename

                with open(filepath, "wb") as f:
                    f.write(img_data)

                registry.add(filename)
                save_registry(registry, NASA_REGISTRY_FILE, NASA_FOLDER)
                downloaded_in_session += 1
                new_added_in_batch += 1

            if new_added_in_batch == 0:
                print("  No new valid images in this batch, requesting another...")

        except Exception as e:
            print(f"  Error querying APOD: {e}")
            break

    print(f"  [APOD Complete] Saved {downloaded_in_session} new images to '{NASA_FOLDER}'.\n")
    return downloaded_in_session


# FAST MAST Archive -> 100 Images
def fetch_mast_images(target_count: int) -> int:
    print(f"Fetching {target_count} High-Res Images from MAST")
    registry = load_registry(MAST_REGISTRY_FILE)
    MAST_FOLDER.mkdir(parents=True, exist_ok=True)

    downloaded_in_session = 0
    now_mjd = Time.now().mjd
    
    # Restringir la búsqueda a los últimos 365 días acelera la consulta 100x
    start_mjd = now_mjd - 365

    try:
        print("  Fast-querying MAST science catalog (HST)...")
        obs = Observations.query_criteria(
            obs_collection="HST",
            dataproduct_type="image",
            intentType="science",
            dataRights="PUBLIC",
            t_min=[start_mjd, now_mjd]
        )

        if len(obs) == 0:
            print("  No recent observations found.")
            return 0

        obs.sort("t_obs_release", reverse=True)
        
        # Procesamos en lotes pequeños de 15 observaciones para que empiece a descargar al instante
        batch_size = 15
        for i in range(0, len(obs), batch_size):
            if len(registry) >= target_count:
                break

            batch_obs = obs[i:i + batch_size]
            products = Observations.get_product_list(batch_obs)
            filtered = Observations.filter_products(
                products,
                extension=["jpg", "png"],
                dataRights="PUBLIC"
            )

            for row in filtered:
                if len(registry) >= target_count:
                    break

                filename = row["productFilename"]
                if filename in registry or "thumb" in filename.lower() or "mini" in filename.lower():
                    continue

                print(f"  [MAST {len(registry)}/{target_count}] Downloading: {filename}...")
                Observations.download_products(
                    filtered[filtered["productFilename"] == filename],
                    download_dir=str(MAST_FOLDER),
                )

                registry.add(filename)
                save_registry(registry, MAST_REGISTRY_FILE, MAST_FOLDER)
                downloaded_in_session += 1

        print(f"  [MAST Complete] Saved {downloaded_in_session} new images to '{MAST_FOLDER}'.\n")
        return downloaded_in_session

    except Exception as e:
        print(f"  Error querying MAST: {e}\n")
        return downloaded_in_session


def main():
    print(f"Starting execution. Goal: {TARGET_PER_SOURCE} images per source.\n")

    # Fetch 100 from NASA APOD
    fetch_apod_images(TARGET_PER_SOURCE)

    # Fetch 100 from MAST
    fetch_mast_images(TARGET_PER_SOURCE)

    print("All Downloads Completed Successfully")


if __name__ == "__main__":
    main()
