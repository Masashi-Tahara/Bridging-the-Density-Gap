# Bridging the Density Gap: Diffusion Model for Stepwise Generation of Dense Cell Images from Sparse Data

Official implementation of the paper accepted at **ISBI 2026**.

**Masashi Tahara, Kazuya Nishimura, Shumpei Takezaki, Ryoma Bise**

## Overview

This repository provides the training and inference code for a diffusion-model-based method that generates dense cell images from sparse data in a stepwise manner.

## Requirements

- NVIDIA GPU with CUDA 12.1
- Docker (recommended)

## Setup

### Using Docker (recommended)

```bash
# Build the image
bash build.sh

# Start the container
bash run.sh

# Attach to the running container
bash exec.sh
```

### Manual installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Dataset

This project uses the C2C12 dataset ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6233481/), [data](https://osf.io/ysaq2/overview)), a fluorescence microscopy dataset of C2C12 myoblast cells with cell position annotations.

Download the dataset and organize the preprocessed data as follows:

```
dataset/
笏懌楳笏 image/       # cell image patches (256x256)
笏披楳笏 heatmap/     # Gaussian heatmap patches (256x256)
```

### Preprocessing

**Step 1: Generate heatmaps**

For each full-resolution image, create a heatmap of the same size where each annotated cell position is represented as a Gaussian blob.
Refer to the `create_heatmap` function in `scripts/segmentation_sample.py` as a reference implementation.

**Step 2: Split images and heatmaps into patches**

Use the provided `split_images.py` to divide both the cell images and heatmaps into overlapping 256ﾃ・56 patches:

```bash
python split_images.py
```

Edit `input_folder` and `output_folder` at the bottom of the script to match your paths.

## Training

```bash
cd scripts
python segmentation_train.py \
    --image_dir <path/to/images> \
    --heatmap_dir <path/to/heatmaps> \
    --save_dir <path/to/output>
```

## Inference

```bash
cd scripts
python segmentation_sample.py \
    --image_dir <path/to/images> \
    --heatmap_dir <path/to/heatmaps> \
    --model_path <path/to/model.pt>
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{tahara2026bridging,
  title     = {Bridging the Density Gap: Diffusion Model for Stepwise Generation of Dense Cell Images from Sparse Data},
  author    = {Tahara, Masashi and Nishimura, Kazuya and Takezaki, Shumpei and Bise, Ryoma},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
}
```

## Acknowledgements

This code is built upon [MedSegDiff](https://github.com/ImprintLab/MedSegDiff).
