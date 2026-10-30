# MATA: Motion Aligned Temporal Adapter
This repository implements **MATA (Motion Aligned Temporal Adapter)**, an adapter for few-shot fine-grained action recognition, accepted in WACV 2027.


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
