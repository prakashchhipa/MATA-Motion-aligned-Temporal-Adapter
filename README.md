# MATA: Motion Aligned Temporal Adapter
A parameter-efficient and practical adapter for few-shot fine-grained action recognition, accepted in WACV 2027.

**Ashish Kumar Singh**<sup>1\*</sup> &nbsp;·&nbsp; **Prakash Chandra Chhipa**<sup>2\*</sup>

<sup>1</sup>Machine Learning Group, Luleå University of Technology &nbsp;&nbsp; <sup>2</sup>Scalable Systems, RISE Research Institutes of Sweden

<sup>\*</sup><i>Joint first authors with equal contributions</i>

<!-- TODO: replace xxxx.xxxxx with the arXiv id, and add a project-page badge if you have one -->
[![WACV 2027](https://img.shields.io/badge/badge/WACV_2027-blue?style=for-the-badge)](https://wacv.thecvf.com/Conferences/2027/index.html)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-CLIP_ViT--B%2F16-38c6f4?style=for-the-badge)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-Apache_2.0-9be15d?style=for-the-badge)](LICENSE)


## Installation

```bash

conda create -n mata python=3.10 -y
conda activate mata
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install other dependencies
pip install -r requirements.txt
```


## Training and Testing

The training and testing are performed using the `runs/run.py` script with dataset-specific configuration files. The script accepts a `--cfg` argument pointing to the configuration YAML file.

### General Command Format

```bash
cd project_folder/
python runs/run.py --cfg <path_to_config_file>
```

### Dataset-Specific Commands

#### Diving48

**Training:**
```bash
python runs/run.py --cfg ../config/diving48/ViT_diving48_d2st_8frames_60kepi.yaml
```

**Testing:**
```bash
# Update the config file to set TRAIN.ENABLE=false and TEST.ENABLE=true
python runs/run.py --cfg ../config/diving48/ViT_diving48_d2st_8frames_60kepi.yaml
```

### Configuration Notes

- Set `TRAIN.ENABLE=true` in the config file for training
- Set `TEST.ENABLE=true` in the config file for testing
- Update `DATA.DATA_ROOT_DIR` and `DATA.ANNO_DIR` in the config file to point to your dataset paths
- Adjust `TRAIN.CHECKPOINT_FILE_PATH` for resuming training or loading pretrained models
- Modify `TEST.CHECKPOINT_FILE_PATH` to specify the model checkpoint for testing

## Visualization and Qualitative Analysis

The `vizualizations/` directory contains tools for three types of qualitative analysis to understand the temporal modeling behavior of MATA:

### 1. Temporal Saliency Analysis

This analysis measures temporal saliency via the per-frame magnitude of adapter-induced residual updates. It shows which temporal frames are most important for the adapter's decision-making process.

**Usage:**
```bash
cd vizualizations
python compute_temporal_saliency.py \
    --cfg_mata <path_to_mata_config> \
    --cfg_d2st <path_to_d2st_config> \
    --output_dir <output_directory> \
    --max_samples <number_of_samples>
```

**Interpretation:**
- Lower magnitude with sharp peaks indicates more efficient and targeted feature adaptation
- Sharp peaks at specific frames show that the adapter knows when to activate at discriminative temporal phases
- See `vizualizations/ADAPTER_MAGNITUDE_INTERPRETATION.md` for detailed interpretation guidelines

### 2. Single-Frame Blackout Sensitivity Analysis

To understand temporal selectivity, we perform single-frame blackout sensitivity analysis on 8-frame sequences. This analysis shows how the model's prediction changes when individual frames are removed.

**Usage:**
```bash
cd vizualizations
python frame_drop_sensitivity.py \
    --cfg_mata <path_to_mata_config> \
    --cfg_d2st <path_to_d2st_config> \
    --output_dir <output_directory> \
    --num_episodes <number_of_episodes> \
    --mode aggregate
```

**Interpretation:**
- Sharp peaks at specific frames indicate high temporal selectivity
- Low baseline values show that the model doesn't need all frames equally
- Clear distinction between peak frames and others demonstrates the model's ability to identify critical temporal moments
- See `vizualizations/HOW_TO_INTERPRET_FRAME_DROP.md` for detailed interpretation guidelines

### 3. Nearest-Neighbor Retrieval Across Confusing Fine-Grained Classes

This analysis computes nearest-neighbor retrieval to understand how well the model distinguishes between confusing fine-grained action classes.

**Usage:**
```bash
cd vizualizations
python nn_retrieval_analysis.py \
    --cfg_mata <path_to_mata_config> \
    --cfg_d2st <path_to_d2st_config> \
    --output_dir <output_directory>
```

**Interpretation:**
- Higher retrieval accuracy for confusing classes indicates better fine-grained discrimination
- Visual inspection of retrieved neighbors shows the model's understanding of class boundaries
