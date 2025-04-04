# MCI Classification

<p align="left">
  <img src="mci.jpeg" width="200" alt="MCI Classification Example"/>
</p>

## Overview

We present the MCI classification training and inference code for BrainIAC as a downstream task. The pipeline is trained and infered on T1 scans, with AUC and F1 as evaluation metric.

## Data Requirements

- **Input**: T1-weighted MR scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped, histogram normalized (optional)
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,1    # 1 for MCI, 0 for healthy control
  ```
refer to [ quickstart.ipynb](../quickstart.ipynb) to find how to preprocess data and generate csv file.

## Setup

1. **Configuration**:
change the [config.yml](../config.yml) file accordingly.
   ```yaml
   # config.yml
   data:
     train_csv: "path/to/train.csv"
     val_csv: "path/to/val.csv"
     test_csv: "path/to/test.csv"
     root_dir: "../data/sample/processed"
     collate: 1  # single scan framework
    
   checkpoints: "./checkpoints/mci_model.00"     # for inference/testing 
   
   train:
    finetune: 'yes'      # yes to finetune the entire model 
    freeze: 'no'         # yes to freeze the resnet backbone 
    weights: ./checkpoints/brainiac.ckpt  # path to brainiac weights
   ```

2. **Training**:
   ```bash
   python -m MCIclassification.train_mci
   ```

3. **Inference**:
   ```bash
   python -m MCIclassification.infer_mci
   ```
