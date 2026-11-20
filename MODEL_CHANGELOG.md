# ASTROCLASS.AI CHANGELOG

> **NOTE**  
> This is a changelog that contains all released versions of the actual Astroclass.ai model, but does not include all commits made in the repository.

## AstroClass 1.2 Flash

### Large-Scale Streaming Fine-Tuning (no local image downloads)
- **What's New & Improvements**
    - **Streaming training pipeline** (`training/stream_train.py`, `training/streaming/`): images are fetched from remote archives, reduced to 256x256 and kept only in RAM as compressed JPEG. Nothing is written to disk.
    - **~70,500 streamed images** from spectroscopically/catalog-confirmed surveys (SDSS DR18, DECaLS via Hugging Face) plus curated press imagery (NASA Images API, ESA/Hubble, Wikimedia Commons). The target of 20,000 per class was reached for `star`, `galaxy` and (almost) `quasar`; `planet` and especially `nebula` are limited by what the reachable archives hold and are smaller (see the table below).
    - **Cleaner data**: word-bounded metadata filters, cross-source near-duplicate removal (dHash over the 8 flips/rotations + low-frequency signature), brightness gate for charts/diagrams, and faint isolated SDSS stars so brightness alone cannot separate `star` from `quasar`.
    - **Astro-aware GPU augmentation**: free rotation, flips, zoom, gamma/contrast stretch, noise and blur; AdamW + label smoothing + EMA weights; class-balanced, source-weighted sampling.
    - **Honest evaluation**: held-out streamed split with confidence interval, curated-only accuracy, source-balanced recall, per-source breakdown and a side-by-side comparison with v1.1.0 on the same images.
- **Trained & Refined**
    - Same architecture and class order as v1.0/v1.1.0 (ResNet-34, 5 classes), so it is a drop-in replacement.
    - Initialised from the repository's `training/trained_net.pth`. On inspection that file had degraded (near-uniform outputs, mean max-probability 0.29, predicts `galaxy` for ~90% of images), so this run is effectively a full retraining on the new data. The original v1.0 weights (`trained_net.pth.backup`) still behave correctly and are used as the baseline below.
    - 23,601 optimisation steps (batch 128, ~3.0M samples seen, ~120 min on an RTX 4060 Laptop GPU), AdamW (lr 2e-4, head 4e-4, weight decay 0.02), label smoothing 0.05, bf16 autocast, EMA weights (decay 0.9985), cosine decay over the last 5,000 steps.
    - Output weights: `training/trained_net_stream.pth`. Run artefacts (log, metrics, learning curves, confusion matrix, dataset summary, held-out and real-image evaluations): `training/stream_runs/20261107_211236/`.
- **Training Data (streamed, RAM only)**

    | Class | Images | Sources |
    | :--- | ---: | :--- |
    | `star` | 20,250 | SDSS spectroscopic stars (bright, crowded-field and faint isolated), NASA, ESA/Hubble |
    | `galaxy` | 21,293 | DECaLS (11,000), SDSS (7,034), Wikimedia Commons, NASA, ESA/Hubble |
    | `quasar` | 18,361 | SDSS spectroscopic QSOs (18,332), NASA, ESA/Hubble |
    | `planet` | 8,509 | NASA Images API (Solar System bodies), ESA/Hubble |
    | `nebula` | 2,128 | Wikimedia Commons (1,794), NASA (234), ESA/Hubble (100) |

    - 5% of the unique images (3,620) were held out before training by a hash of the image id.
    - Removed during streaming: 7,974 near-duplicates, 6,253 cross-class conflicts, 4,163 unusable frames (blank, bright charts, odd aspect ratio).
- **Performance & Metrics**
    - **Held-out streamed images** (5% split never used for training, n = 4,131 re-streamed from the same sources, ESA excluded):

        | Model | Accuracy | Balanced acc. | Curated-only acc. (NASA/Commons, n = 898) |
        | :--- | ---: | ---: | ---: |
        | **1.2 Flash** | **93.4%** | **93.8%** | **97.0%** |
        | v1.0 (`trained_net.pth.backup`) | 36.7% | 45.9% | 75.1% |
        | previous `trained_net.pth` (degraded) | 25.6% | 22.3% | 22.3% |

        1.2 Flash per-class recall: `star` 90.6%, `galaxy` 97.2%, `quasar` 89.9%, `nebula` 91.7%, `planet` 99.5%. The 95% confidence interval of the training-run evaluation (n = 3,620, TTA) is 93.1-94.7%. v1.0 scores low mainly because it never saw SDSS-style `star` and `quasar` cutouts (recall 1.6% and 2.1%).
    - **Real-world images** (103 pictures in `images/nasaDownloads`, each labelled independently by three reviewers, 97 unanimous; 18 images fitting none of the five classes were excluded, leaving 85):

        | Model | Accuracy |
        | :--- | ---: |
        | **1.2 Flash** | **87.1%** (74/85) |
        | v1.0 (`trained_net.pth.backup`) | 71.8% (61/85) |
        | previous `trained_net.pth` (degraded) | 38.8% (33/85) |

        Per class (1.2 Flash): `nebula` 37/38, `galaxy` 25/27, `planet` 10/11, `star` 2/6, `quasar` 0/3.
- **Known Limitations**
    - **Globular clusters are often classified as `galaxy`** (e.g. 47 Tucanae, M10, NGC 6366): `star` training data is mostly SDSS cutouts, with only ~150 curated cluster/field images.
    - **Real quasars, pulsars and jets are not recognised** (M87 jet, Vela pulsar, PSR J0030 in the real-image check): `quasar` is almost entirely SDSS point sources, which look like stars. On SDSS data `star` vs `quasar` is the main confusion (104 stars predicted as quasar, 46 quasars as star in the training-run evaluation).
    - **`nebula` is the smallest class** (2,128 images, ~100 held-out): its recall estimate is noisy.
    - **Non-class images are forced into one of the five classes**: Moon, Milky Way landscapes, comets, meteor showers and protoplanetary discs are not handled by an "other" class.
    - Real-image labels come from independent reviewers (not verified by astronomers) and `star` (6) and `quasar` (3) have very few samples there, so those two percentages are very noisy.
    - Not yet published to Hugging Face.


## AstroClass.ai v1.1.0

### In-Process Active Supervision & Weight Refinement
- **What's New & Improvements**
    - **Active Online Weight Updates**: Refined ResNet-34 weights through in-process Qwen 2.5 AI validation, running 100% locally with 4-bit quantization on GPU without external services or APIs.
    - **Astrophysical Feature Extraction**: Integrated photometric and morphological descriptors (mean luminance, dark space background ratio, RGB channel dynamics, core concentration index, and edge contrast) to guide auditing.
    - **Adaptive Confidence-Scaled Backpropagation**: Executed targeted gradient descent steps scaled dynamically by Qwen's assessment confidence level (`high`, `medium`, `low`).
    - **BatchNorm Running Statistics Freezing**: Froze BatchNorm layers during single-sample online training steps to prevent covariate shift and preserve global model stability.
    - **Refined Decision Boundaries**: Significantly improved distinction on challenging astronomical edge cases, such as separating diffuse planetary nebulae from faint spiral galaxies, and isolated stellar cores from sensor artifacts.
    - **Hugging Face Hub Distribution**: Packaged and published weights (`astroclass_ai.pth`), model card, and version tags directly to Hugging Face Hub (`RaulSalasSahuquillo/astroclass_ai`).
- **Trained & Refined**
    - Built upon the baseline 25-epoch weights with 203 adaptive online gradient descent steps.
    - Supervised locally using `Qwen2.5-Coder-7B-Instruct` (in-process via Hugging Face `transformers` and `bitsandbytes`).
    - Optimized with Adam at a conservative learning rate (`lr = 1e-4`) for fine-grained boundary refinement.
- **Performance & Metrics**
    - Test classification accuracy elevated to **94.2%**.
    - Fast sub-second inference runtime preserved for local and cloud environments.


## AstroClass.ai v1.0.0

### New AI model that classifies your images
- **Does**
    - Get images from open libraries such as NASA's and downloads it into the local or virtual environment you're executing the AI in.
    - Generates an entire dataset with csv that contains all the avaliable data about each image.
    - Gives the .pth model the images and the access to the environment's folders.
    - Analizes the image and gives a final response within 5 seconds.
    - Sorts the image in the classified folders.
- **Trained**
    - With 25 epochs
    - 5000 classifies images from NASA and MAST (1000 each group)
    - With the Google Colab's GPU T4