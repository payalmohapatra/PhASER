<div align="center">
  <h2 align="center">Phase-driven Domain Generalizable Learning for Nonstationary Time Series</h2>
  
  <div>
    <a href="https://arxiv.org/abs/2402.05960" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.05960-b31b1b.svg?style=flat">
    </a>
    <a href="https://www.jmlr.org/papers/v24/23-1234.html" target="_blank">
      <img alt="TMLR" src="https://img.shields.io/badge/TMLR-Accepted-brightgreen.svg">
    </a>
    <a href="#" target="_blank">
      <img alt="NeurIPS Workshop" src="https://img.shields.io/badge/TS4H%20Workshop-NeurIPS%202025-orange.svg">
    </a>
    <a href="https://opensource.org/licenses/MIT" target="_blank">
      <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
  </div>
  <img src="https://github.com/payalmohapatra/PhASER/blob/main/render/TMLRMainTSDGRevamp.drawio.png" alt="System Diagram" width="80%">
<!-- ![System_Diagram](https://github.com/payalmohapatra/PhASER/blob/main/render/NIPsMainTSDGRevamp.png) -->
  
  <p>
    <strong>Addressing domain generalization challenges in time series analysis through phase-driven learning</strong>
  </p>
</div>



Here we provide all the required scripts to reproduce the results from the last row of Tables 2, 3 and 4 in the main paper. These will demonstrate the performance of PhASER on 5 datasets.

# Requirement 

The required packages are listed in requirements.txt and can be installed as :
```
pip install -r requirements.txt
```

If you want to create a new Conda environment, you can also run the following:

```
conda env create -f environment.yml
```
# Code Organisation
./util/dataset_cfg.py -- Contains all the dataset configurations. Dataset specific hyperparameters like STFT specifications, batch size, out of domain scenarios' splits, etc. can be defined here.

./phaser_models.py -- Contains building blocks for CNNS and sub-feature normalisation.

./PhASER_XXX.py -- Complete scripts for HAR, EMG and EEG applications. Note that the dataset formats are slightly different, so the dataloader is manipulated accordingly. For any new datasets, the dataloader may need to be revisited.

At the end of the training a new directory ./model_chkpt/. will be created to save the trained models. A *_acc.csv to report the validation and target accuracy will also be generated.

# Dataset
The processed data can be place here - PhASER_ICML_CodeBase/processed_data or the absolute path for the dataset can be given as an argument.


The dataset can be obtained at the following sources : <br>
1. [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)
2. [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
3. [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
4. [Gesture Recognition](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures)
5. [Sleep-Stage Classification](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)


# 
# Training scripts
The main scripts to train a PhASER can be run using the following command to reproduce results in the Tables 2-4. The args.scenario and args.seed_num can be updated for the various combinations. args.scenario defines the various out-of-domain settings. After downloading the respective datasets, the path can be provided using args.dataset_pth.

## WISDM
```
python PhASER_WISDM.py  --dataset_pth='/processed_data/WISDM/' --scenario='S1' --seed_num=2711 --num_epochs=30

```
## HHAR

### Cross Person setting
```
python PhASER_HHAR.py --dataset_pth='/processed_data/HHAR/' --scenario='S1' --seed_num=2711 --num_epochs=30
```
### One-to-Another setting
For the cross person setting update args.oot (instead of args.scenario) to train the model using various single person domains.
```
python PhASER_HHAR.py --dataset_pth='/processed_data/HHAR/' --oot=0 --seed_num=2711 --num_epochs=30 
```


## UCIHAR
```
python PhASER_UCIHAR.py --dataset_pth='/processed_data/UCIHAR/' --scenario='S1' --seed_num=2711 --num_epochs=30
```

## GR
```
python PhASER_EMG.py --seed_num=2711 --scenario='S1' --num_epochs=50 
```

## SSC
```
python PhASER_EEG.py --seed_num=2711 --scenario='S1' --num_epochs=20 
```


# Results
All the results are written as a *.csv file at the end for offline analyses.


# Citation
```
@article{mohapatra2025phaser,
  title={Phase-driven domain generalizable learning for nonstationary time series},
  author={Mohapatra, Payal and Wang, Lixu and Zhu, Qi},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2025},
  url = {https://openreview.net/forum?id=cb3nwoqLdd}
}
```

# Acknowledgement 
We would like to thank the [AdATIMe benchmarking suite](https://github.com/emadeldeen24/AdaTime) for providing the domain-generalization data and processing utilities used in this project.