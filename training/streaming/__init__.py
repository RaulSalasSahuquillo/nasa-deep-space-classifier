"""Streaming (in-memory only) dataset tooling for AstroClass.ai training.

Images are fetched from remote archives, decoded, normalised to 256x256 and kept
as compressed JPEG bytes in RAM. Nothing is ever written to disk.
"""

CLASS_NAMES = ["star", "galaxy", "quasar", "nebula", "planet"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
