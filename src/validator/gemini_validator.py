#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# Author:       Raúl Salas Sahuquillo
# Repository:   https://github.com/RaulSalasSahuquillo/nasa-image-training
# License:      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
# File:         src/validator/gemini_validator.py
# Description:  Backwards-compatible wrapper delegating to local In-Process Qwen AI
#               Validator & ResNet-34 Active Neural Trainer (src/validator/qwen_validator.py).
#               Maintains identical CLI arguments, outputs, and workflows.
# ==============================================================================

import sys
from pathlib import Path

# Add validator directory to path if needed
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_validator import main

if __name__ == "__main__":
    main()
