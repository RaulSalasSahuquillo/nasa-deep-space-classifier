# `security-policy`

> *"Initializing security auditing subsystem..."*  
> `[OK] Secret detection filters active.`  
> `[OK] Dependency vulnerability monitoring enabled.`  
> `[OK] Security policies enforceable.`  

The **nasa-image-training** project takes security, data integrity, and responsible disclosure seriously. This document outlines our security policies, supported versions, secret management practices, and vulnerability reporting procedures.

---

### `/usr/bin/security_policy`

We maintain active security updates for the following components:

| Component | Target Scope | Security Level | Status |
| :--- | :--- | :--- | :--- |
| **`src/viewer/`**<br><br>![](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square) | Local HTTP Server & SQLite Connection | Input sanitization, SQL Injection prevention, local network exposure. | :white_check_mark: Supported |
| **`src/data_collection/`**<br><br>![](https://img.shields.io/badge/API-NASA%20%26%20MAST-E03C31?style=flat-square) | HTTPS Requests & Asset Downloads | Safe payload handling, timeout protection, filename sanitization. | :white_check_mark: Supported |
| **`training/`**<br><br>![](https://img.shields.io/badge/PyTorch-Weights-EE4C2C?style=flat-square) | Model Artifacts & CSV Data | Checksum validation, safe `torch.load` deserialization with weights verification. | :white_check_mark: Supported |

---

### `/etc/secret_management`

#### 1. NASA API Keys & Tokens
* Never hardcode API keys or credentials directly into Python files, Jupyter Notebooks, or commits.
* Store keys in `.env` (which is ignored by [`.gitignore`](.gitignore)).
* Use [`.env.example`](.env.example) as the single source of truth for required environment variables.

#### 2. Model Weights & Pickles
* Avoid loading `.pth` or pickle files from untrusted third-party origins.
* Always specify safe execution modes (`weights_only=True` when applicable in newer PyTorch releases).

#### 3. Local Web Server Exposure
* By default, `src/viewer/classification_viewer.py` runs in local development mode. Do not expose development servers directly to untrusted public networks without reverse proxies or proper authentication.

---

### `/var/log/reporting_a_vulnerability`

If you discover a potential security flaw, credential leak, or vulnerability in this repository, please follow our coordinated disclosure process:

1. **Do NOT file a public GitHub issue.**
2. Send a detailed report via email to **Raúl Salas Sahuquillo** or use GitHub's private security advisory feature:
   * **Subject:** `[SECURITY] Vulnerability Report - nasa-image-training`
   * **Details:** Provide a step-by-step reproduction guide, potential impact, and suggested remediations.
3. You will receive an acknowledgment within **48 hours**.
4. Once verified, a patch will be prepared and published alongside an advisory release.

---

*"Security is not a feature, it is an ongoing celestial watch."*

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0) license.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### Summary of Terms:
* **Attribution:** You must give appropriate credit to **Raúl Salas Sahuquillo** and provide a link to the original repository.
* **Non-Commercial:** You may not use this work, its code, or trained model weights for commercial purposes without explicit written consent.
* **ShareAlike:** If you adapt or build upon this repository, your modifications must be shared under the same license.
