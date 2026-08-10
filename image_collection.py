#!/usr/bin/env python3
"""
image_collection.py
--------------------
Downloads public "cosmos" images from MAST (Mikulski Archive for Space
Telescopes) using the astroquery library.

Program flow:
    1. On startup, it looks up the 10 most recent public images and
       downloads any that aren't already in the project's "images" folder.
    2. After that initial download, it runs ONE extra check for any new
       image that may have appeared in the meantime, and downloads it.
       (No loop, no scheduling — it runs once and exits.)

Requirements:
    pip install astroquery astropy

Usage:
    python image_collection.py
"""

import json
from pathlib import Path
from astropy.time import Time
from astroquery.mast import Observations

# Configuration
IMAGES_FOLDER = Path("./images")                        # destination folder
REGISTRY_FILE = IMAGES_FOLDER / "downloaded.json"       # tracks what's already downloaded
NUM_IMAGES = 10                                         # how many recent images to track
MISSIONS = ["JWST", "HST"]                              # missions to query
IMAGE_EXTENSIONS = ["jpg", "png"]                       # preview images only, not raw FITS

# Searching the ENTIRE archive (millions of HST/JWST observations) and
# sorting locally is very slow. Instead we search within a recent time
# window, and widen it only if that's not enough to find NUM_IMAGES.
INITIAL_WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 3650  # don't search back more than ~10 years


# Local registry of already-downloaded files so we don't re-download
def load_registry() -> set:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_registry(registry: set) -> None:
    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(sorted(registry), f, indent=2)


# MAST queries
def find_recent_images(n: int):
    """Returns the n most recent, public 'image' observations.

    Instead of pulling the entire archive, this searches within a recent
    time window (t_min = observation start time, in MJD) and widens that
    window only if it doesn't turn up enough results. This keeps queries
    fast (seconds) instead of scanning the whole catalog.
    """
    now_mjd = Time.now().mjd
    window_days = INITIAL_WINDOW_DAYS
    obs = []

    while True:
        start_mjd = now_mjd - window_days
        print(f"  Searching the last {window_days} day(s)...")

        obs = Observations.query_criteria(
            obs_collection=MISSIONS,
            dataproduct_type="image",
            dataRights="PUBLIC",
            t_min=[start_mjd, now_mjd],
        )

        if len(obs) >= n or window_days >= MAX_WINDOW_DAYS:
            break

        window_days *= 4  # widen the search and try again

    if len(obs) == 0:
        return obs

    # t_obs_release = date (in MJD) the data was made public.
    # Sort from most recent to oldest.
    obs.sort("t_obs_release", reverse=True)
    return obs[:n]


def get_downloadable_products(obs):
    """From those observations' products, keep only public image
    previews (jpg/png).

    Note: preview images (jpg/png) are typically tagged productType
    "PREVIEW" or "THUMBNAIL" in MAST, not "SCIENCE" (SCIENCE is for the
    raw/calibrated FITS data). So we filter only by extension and
    public data rights, without restricting productType.
    """
    if len(obs) == 0:
        return None

    products = Observations.get_product_list(obs)
    filtered_products = Observations.filter_products(
        products,
        extension=IMAGE_EXTENSIONS,
        dataRights="PUBLIC",
    )
    return filtered_products


def download_new(products, registry: set) -> int:
    """Downloads only the products whose filename isn't already in the
    registry. Returns how many were downloaded."""
    if products is None or len(products) == 0:
        return 0

    new_count = 0
    for row in products:
        filename = row["productFilename"]
        if filename in registry:
            continue  # already have it

        Observations.download_products(
            products[products["productFilename"] == filename],
            download_dir=str(IMAGES_FOLDER),
        )
        registry.add(filename)
        new_count += 1

    return new_count


def main():
    registry = load_registry()

    # Initial download of the N most recent public images
    print(f"Looking up the {NUM_IMAGES} most recent public images on MAST...")
    initial_obs = find_recent_images(NUM_IMAGES)

    if len(initial_obs) == 0:
        print("No observations found. Check your connection or the MISSIONS filter.")
        return

    products = get_downloadable_products(initial_obs)
    if products is not None:
        print(f"  Found {len(initial_obs)} observation(s), {len(products)} downloadable image file(s).")
    downloaded = download_new(products, registry)
    save_registry(registry)
    print(f"Initial download complete: {downloaded} new image(s) saved to '{IMAGES_FOLDER}'.")

    # One-time check for new images
    print("\nChecking once for any new image since the initial download...")
    current_obs = find_recent_images(NUM_IMAGES)
    current_products = get_downloadable_products(current_obs)
    extra_downloaded = download_new(current_products, registry)
    save_registry(registry)

    if extra_downloaded > 0:
        print(f"Found and downloaded {extra_downloaded} new image(s).")
    else:
        print("No new images found.")

    print("\nDone. The program finished.")


if __name__ == "__main__":
    main()