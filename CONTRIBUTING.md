# `contributing-guide`

> *"Initializing contributor onboarding protocol..."*  
> `[OK] Git development workflow configured.`  
> `[OK] Code style and typing linters loaded.`  
> `[OK] Ready to accept contributions.`  

Thank you for your interest in contributing to **nasa-image-training**! We welcome contributions ranging from dataset additions and model optimizations to web dashboard enhancements, architectural refactors, and documentation improvements.

---

### `/usr/bin/contribution_rules`

To ensure smooth collaboration, please review our core contribution principles:

| Rule | Area | Description |
| :--- | :--- | :--- |
| **[Non-Commercial](LICENSE)**<br><br>![](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square) | **Licensing** | All contributions fall under the **CC BY-NC-SA 4.0** license. No commercial use or closed-source redistribution. |
| **[Clean Code](src/)**<br><br>![](https://img.shields.io/badge/Python-PEP%208-3776AB?style=flat-square&logo=python&logoColor=white) ![](https://img.shields.io/badge/Linter-Flake8%20%7C%20Black-black?style=flat-square) | **Code Quality** | Write PEP 8 compliant, well-documented Python code. Keep programs organized inside [`src/`](src/). |
| **[Reproducibility](training/)**<br><br>![](https://img.shields.io/badge/PyTorch-Deterministic-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | **Deep Learning** | Ensure dataset splits, seeds, and training hyperparameters are documented and reproducible. |
| **[No Plain Secrets](.env.example)**<br><br>![](https://img.shields.io/badge/Security-No%20API%20Keys-red?style=flat-square) | **Secrets** | Never commit real NASA API keys or private tokens. Always use `.env` and reference `.env.example`. |

---

### `/usr/bin/git_workflow`

Follow these standard steps to submit your contributions:

#### 1. Fork & Clone the Repository
```bash
git clone https://github.com/<your-username>/nasa-image-training.git
cd nasa-image-training
```

#### 2. Create a Feature Branch
```bash
git checkout -b feature/astronomical-target-expansion
# or: git checkout -b fix/harvester-retry-timeout
```

#### 3. Set Up Local Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. Make Your Changes
- Ensure all Python modules are structured under [`src/`](src/) subpackages.
- Keep HTML views clean, accessible, and free of unnecessary CSS frameworks.
- Test data collectors, inference scripts, or web servers locally before committing.

#### 5. Commit and Push
Use clear, imperative commit messages (following Conventional Commits):
```bash
git commit -m "feat(harvester): add retry exponential backoff for MAST queries"
git push origin feature/astronomical-target-expansion
```

#### 6. Open a Pull Request
- Provide a concise summary of what was changed and why.
- Reference any related issues (e.g. `Resolves #12`).
- Verify that your code adheres to the project license.

---

### `/var/log/areas_to_explore`

Looking for ideas to contribute? Here are key areas open for enhancement:

* **Data Ingestion:** Add support for James Webb Space Telescope (JWST) MAST queries or ESO archive images.
* **Model Architecture:** Experiment with Vision Transformers (ViT) or ConvNeXt backends.
* **Metrics & Evaluation:** Add confusion matrix generation and per-class precision/recall reporting.
* **Web Viewer:** Add pagination, category search filtering, or interactive image metadata previews.

---

*"Great science is built on collective curiosity and collaborative code."*

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0) license.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### Summary of Terms:
* **Attribution:** You must give appropriate credit to **Raúl Salas Sahuquillo** and provide a link to the original repository.
* **Non-Commercial:** You may not use this work, its code, or trained model weights for commercial purposes without explicit written consent.
* **ShareAlike:** If you adapt or build upon this repository, your modifications must be shared under the same license.
