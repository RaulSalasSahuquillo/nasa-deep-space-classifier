# ASTROCLASS.AI CHANGELOG

> **NOTE**  
> This is a changelog that contains all released versions of the actual Astroclass.ai model, but does not include all commits made in the repository.

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