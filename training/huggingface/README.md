---
language:
- en
license: cc-by-nc-sa-4.0
tags:
- astronomy
- astrophysics
- deep-space
- space-telescope
- hubble
- nasa
- mast
- image-classification
- resnet
- pytorch
- computer-vision
- multimodal-audit
datasets:
- custom
metrics:
- accuracy
pipeline_tag: image-classification
library_name: generic
model-index:
- name: astroclass_ai
  results:
  - task:
      type: image-classification
      name: Astronomical Deep Space Image Classification
    metrics:
    - type: accuracy
      value: 93.4
      name: Held-out streamed accuracy (1.2 Flash)
---

# `astroclass_ai` (1.2 Flash) — NASA Deep Space Image Classifier

> **Astrophysical Deep Learning Backbone (ResNet-34)** trained on ~70,500 images streamed from SDSS, DECaLS, the NASA Images API, ESA/Hubble and Wikimedia Commons (1.2 Flash), on top of the observational data from NASA Astronomy Picture of the Day (APOD) and the Mikulski Archive for Space Telescopes (MAST Hubble Space Telescope) used by earlier versions, with a local **Qwen 2.5** validator for auditing.

* **Version:** `1.2 Flash` (tag `v1.2-flash`, large-scale streaming fine-tune)
* **Author:** Raúl Salas Sahuquillo
* **GitHub Repository:** [RaulSalasSahuquillo/nasa-image-training](https://github.com/RaulSalasSahuquillo/nasa-image-training)
* **Project Web Portal:** [NASA Deep Space Classifier Web](https://raulsalassahuquillo.github.io/nasa-deep-space-classifier-website/)
* **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

---

## Release Notes & Versions

| Version | Status | Model Weights (`astroclass_ai.pth`) | Description |
| :---: | :---: | :---: | :--- |
| **`1.2 Flash`** (tag `v1.2-flash`) | **Latest** | **Updated** (streaming fine-tune, 23.6k steps, EMA weights) | Trained on ~70,500 streamed images (star 20,250 / galaxy 21,293 / quasar 18,361 / planet 8,509 / nebula 2,128) with word-bounded label filters, near-duplicate removal and a held-out evaluation. 93.4% accuracy on held-out streamed images and 87.1% on 85 independently labelled real-world pictures (v1.0: 71.8%). See *Evaluation & Known Limitations* below. |
| **`v1.1.0`** | Previous | **Updated** (Qwen 2.5 In-Process Supervised) | Model weights refined with 203 adaptive online gradient descent steps guided by local Qwen 2.5 astronomical evaluation and astrophysical feature extraction. Enhanced distinction of stellar cores, faint nebular structures, and sensor artifacts. |
| **`v1.0.0`** | Historical | Baseline (25 Epochs) | Initial release trained on APOD and MAST Hubble Space Telescope imagery using ResNet-34 transfer learning. |

---

## Model Description

`astroclass_ai` is a deep learning model based on the **ResNet-34** architecture, trained to identify and categorize deep space optical and ultraviolet captures into five core astrophysical classes:

| Class Index | Target Class | Astronomical Morphology & Scope |
| :---: | :--- | :--- |
| `0` | **`star`** | Single stars, globular clusters, open clusters, stellar fields, and star trails. |
| `1` | **`galaxy`** | Spiral, elliptical, lenticular, irregular, merging galaxies, and galaxy clusters. |
| `2` | **`quasar`** | Active Galactic Nuclei (AGN), distant pulsars, and high-energy radio cores. |
| `3` | **`nebula`** | Emission, reflection, planetary nebulae, supernova remnants, and dark clouds. |
| `4` | **`planet`** | Solar system planets, planetary satellites, rings, and surface observations. |

---

## Quick Inference with PyTorch & Hugging Face Hub

You can download the model weights directly and run classification on any astronomical image with PyTorch:

```python
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
from huggingface_hub import hf_hub_download

# 1. Download weights from Hugging Face Hub (1.2 Flash)
model_path = hf_hub_download(
    repo_id="RaulSalasSahuquillo/astroclass_ai",
    filename="astroclass_ai.pth",
    revision="main"  # or revision="v1.2-flash"
)

# 2. Instantiate ResNet-34 with 5-class classification head
classes = ["star", "galaxy", "quasar", "nebula", "planet"]
model = resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

# 3. Load weights
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 4. Standard preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Classify an astronomical image
image_path = "space_observation.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_idx = probabilities.argmax().item()

print(f"Target:     {image_path}")
print(f"Prediction: {classes[predicted_idx]} ({probabilities[predicted_idx]*100:.2f}%)")
```

---

## Evaluation & Known Limitations (1.2 Flash)

**Training data (streamed into RAM, never stored on disk):** `star` 20,250 (SDSS spectroscopic stars + NASA/ESA), `galaxy` 21,293 (DECaLS, SDSS, Commons, NASA, ESA), `quasar` 18,361 (SDSS spectroscopic QSOs + NASA/ESA), `planet` 8,509 (NASA Images API), `nebula` 2,128 (Commons, NASA, ESA). `nebula` and `planet` are smaller because the reachable archives hold fewer well-labelled images.

| Evaluation | 1.2 Flash | v1.0 baseline |
| :--- | ---: | ---: |
| Held-out streamed images (n = 4,131) | **93.4%** acc. / 93.8% balanced | 36.7% / 45.9% |
| Held-out, curated sources only (NASA/Commons, n = 898) | **97.0%** | 75.1% |
| 85 real pictures labelled independently by 3 reviewers each | **87.1%** (74/85) | 71.8% (61/85) |

1.2 Flash per-class recall on the held-out set: `star` 90.6%, `galaxy` 97.2%, `quasar` 89.9%, `nebula` 91.7%, `planet` 99.5%. The v1.0 baseline scores low on the held-out set mainly because it never saw SDSS-style `star` and `quasar` cutouts.

**Known limitations**
* Globular clusters (e.g. 47 Tucanae, M10) are often classified as `galaxy`; `star` training data is mostly SDSS cutouts.
* Real quasars, pulsars and jets (M87 jet, Vela pulsar) are not recognised reliably; `quasar` is almost entirely SDSS point sources that look like stars.
* `nebula` has few images (2,128, ~100 held out), so its recall estimate is noisy.
* There is no "other" class: Moon, Milky Way landscapes, comets or meteor showers are forced into one of the five classes.
* The real-picture labels come from independent reviewers (not astronomers) and `star` (6) / `quasar` (3) have very few samples there.

---

## Training Details & Hyperparameters

### 1.2 Flash
* **Architecture:** ResNet-34, 5-class head (unchanged, drop-in replacement).
* **Data:** streamed from SDSS DR18, DECaLS (Hugging Face), NASA Images API, ESA/Hubble and Wikimedia Commons; held-out 5% split by image-id hash; cross-source near-duplicate removal.
* **Optimization:** AdamW (lr 2e-4, head 4e-4, weight decay 0.02), label smoothing 0.05, batch 128, bf16 autocast, EMA weights (0.9985), cosine decay over the last 5,000 of 23,601 steps.
* **Augmentations (GPU):** free rotation, flips, zoom/crop, gamma and contrast stretch, noise, blur.
* **Sampling:** class-balanced, source-weighted.

### v1.0 / v1.1.0

* **Architecture:** ResNet-34 (Transfer Learning with pre-trained ImageNet backbone).
* **Optimization:** Adam (`learning_rate = 1e-4`).
* **Loss Function:** Multi-class Cross-Entropy Loss (`nn.CrossEntropyLoss`).
* **Acceleration:** PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`).
* **Augmentations:** Random horizontal flips, random 180° rotations, and standard image normalization.
* **Epochs:** 25 + Online In-Process Supervisor Gradient Updates.

---

## Local AI Supervision & Active Weight Training (Qwen 2.5)

In version **`v1.1.0`** (and not in 1.2 Flash, which was trained on streamed data instead), the network weights were audited and trained locally using **Qwen 2.5** operating completely in-process on GPU. 

For each observation:
1. **Astrophysical Feature Extraction:** Analysis of mean luminance, background darkness, RGB channel distribution, and central concentration index.
2. **AI Morphology Evaluation:** Qwen acts as an astrophysical expert, verifying or correcting the classification.
3. **Adaptive Online Gradient Descent:** When corrections or confidence adjustments are needed, Qwen actively updates the ResNet-34 weights via single-sample backpropagation with frozen BatchNorm statistics to preserve global stability while refining decision boundaries.

For the full training notebooks, automated data harvester, SQLite logger, Qwen validator/trainer, and web viewer dashboard, visit the official [GitHub Repository](https://github.com/RaulSalasSahuquillo/nasa-image-training).

---

## License

This model and its associated weights are distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0) license.

* **Attribution:** Credit Raúl Salas Sahuquillo and link to the source repository.
* **Non-Commercial:** The model and weights may not be used for commercial purposes without explicit permission.
* **ShareAlike:** Adaptations must be shared under the same license terms.
