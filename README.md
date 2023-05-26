# DCKS
> The official implementation for the paper *Improving Empathetic Dialogue Generation by Dynamically Infusing Commonsense Knowledge*.

<img src="https://img.shields.io/badge/Venue-ACL--2023-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red"> <img src="https://img.shields.io/badge/Last%20Updated-2023--05--24-2D333B" alt="update"/>

## Usage

### Dependencies

Install the required libraries (Python 3.8.5 | CUDA 10.2)

```sh
pip install -r requirements.txt 
```

### Dataset

The preprocessed dataset is already provided, you can download the data from [Baidu Netdisk](https://pan.baidu.com/s/1A0vEm4Yo6DZGSLcyWIfffg) (passcode: **4b8l**) , and then put it in the folder `/data/ED/dataset`.   
If you want to create the dataset yourself, delete this file, download the [COMET checkpoint](https://github.com/allenai/comet-atomic-2020) and place it in `/data`. The preprocessed dataset would be generated after the training script.

### Training

```sh
bash run_train.sh
```

where data_ratio can be used for low resource ablation studies.

### Evaluation

```sh
bash run_evaluate.sh
```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{DCKS2023,
      title={Improving Empathetic Dialogue Generation by Dynamically Infusing Commonsense Knowledge}, 
      author={Hua Cai, Xuli Shen, Qing Xu, Weilin Shen, Xiaomei Wang, Weifeng Ge, Xiaoqing Zheng, Xiangyang Xue},
      journal={arXiv preprint arXiv:2109.05739},
      year={2023},
}
```
