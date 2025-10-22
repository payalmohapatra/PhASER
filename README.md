<div align="center">

<!-- Header with TMLR logo and title -->
<div style="display: flex; align-items: center; justify-content: center; gap: 16px;">
  <img src="render/tmlr_logo.png" alt="TMLR Logo" height="70">
  <h1 style="margin: 0;">
    PhASER: Phase-driven Generalizable Representation Learning for Nonstationary Time Series Classification
  </h1>
</div>

<br>

<div>
  <a href="https://arxiv.org/abs/2402.05960" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.05960-b31b1b.svg?style=flat">
  </a>
  <a href="https://openreview.net/forum?id=cb3nwoqLdd" target="_blank">
    <img alt="TMLR" src="https://img.shields.io/badge/TMLR-Accepted-brightgreen.svg">
  </a>
  <a href="https://timeseries4health.github.io/" target="_blank">
    <img alt="NeurIPS Workshop" src="https://img.shields.io/badge/TS4H%20Workshop-NeurIPS%202025-orange.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</div>

<br>

<img src="https://github.com/payalmohapatra/PhASER/blob/main/render/TMLRMainTSDGRevamp.drawio.png" alt="PhASER Architecture" width="85%">

</div>

<br>

##  Overview

This repository contains the implementation of **PhASER**, demonstrating results from Tables 2, 3, and 4 in our TMLR paper. The code evaluates PhASER's performance across 5 time series datasets for domain generalization tasks.

<br>

##  Quick Start

### Installation
```bash
# Using pip
pip install -r requirements.txt

# Or create a new Conda environment
conda env create -f environment.yml
```

### Project Structure
```
├── util/dataset_cfg.py       # Dataset configurations and hyperparameters
├── phaser_models.py          # CNN building blocks and feature normalization
├── PhASER_*.py               # Application-specific scripts (HAR, EMG, EEG)
└── processed_data/           # Place your datasets here
```

<br>

## Datasets

Download the preprocessed datasets from the following sources:

| Dataset | Domain | Link |
|---------|--------|------|
| **WISDM** | Human Activity Recognition | [Download](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B) |
| **UCIHAR** | Human Activity Recognition | [Download](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ) |
| **HHAR** | Human Activity Recognition | [Download](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO) |
| **GR** | Gesture Recognition (EMG) | [Download](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures) |
| **SSC** | Sleep Stage Classification (EEG) | [Download](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9) |

<br>

## Training

Run the following commands to reproduce results. Update `--scenario` and `--seed_num` for different experimental settings.

### WISDM
```bash
python PhASER_WISDM.py --dataset_pth='./processed_data/WISDM/' --scenario='S1' --seed_num=2711 --num_epochs=30
```

### UCIHAR
```bash
python PhASER_UCIHAR.py --dataset_pth='./processed_data/UCIHAR/' --scenario='S1' --seed_num=2711 --num_epochs=30
```

### HHAR

**Cross-Person Setting:**
```bash
python PhASER_HHAR.py --dataset_pth='./processed_data/HHAR/' --scenario='S1' --seed_num=2711 --num_epochs=30
```

**One-to-Another Setting:**
```bash
python PhASER_HHAR.py --dataset_pth='./processed_data/HHAR/' --oot=0 --seed_num=2711 --num_epochs=30
```

### Gesture Recognition (GR)
```bash
python PhASER_EMG.py --seed_num=2711 --scenario='S1' --num_epochs=50
```

### Sleep Stage Classification (SSC)
```bash
python PhASER_EEG.py --seed_num=2711 --scenario='S1' --num_epochs=20
```

**Output:** Trained models are saved in `./model_chkpt/`, and validation/target accuracies are logged in CSV files.

<br>

## Citation

If you find this work helpful, please cite:
```bibtex
@article{mohapatra2025phaser,
  title={Phase-driven domain generalizable learning for nonstationary time series},
  author={Mohapatra, Payal and Wang, Lixu and Zhu, Qi},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2025},
  url={https://openreview.net/forum?id=cb3nwoqLdd}
}
```

<br>

##  Acknowledgments

We thank the [AdATIMe benchmarking suite](https://github.com/emadeldeen24/AdaTime) for providing the domain-generalization data and processing utilities used in this project.

<br>

