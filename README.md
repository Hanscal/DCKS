# DCKS
> The official implementation for the paper *Improving Empathetic Dialogue Generation by Dynamically Infusing Commonsense Knowledge*.

<img src="https://img.shields.io/badge/Venue-ACL--2023-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red"> <img src="https://img.shields.io/badge/Last%20Updated-2023--05--24-2D333B" alt="update"/>

## Usage

### Dependencies

Install the required libraries (Python 3.8.5 | CUDA 10.2)

```sh
pip install -r requirements.txt 
```

Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

### Dataset

The preprocessed dataset is already provided, you can download the data from [Baidu Netdisk](https://pan.baidu.com/s/1A0vEm4Yo6DZGSLcyWIfffg) (passcode: **4b8l**) , and then put it in the folder `/data/ED/dataset`.   
If you want to create the dataset yourself, delete this file, download the [COMET checkpoint](https://github.com/allenai/comet-atomic-2020) and place it in `/data/ed_comet`. The preprocessed dataset would be generated after the training script.

### Training

```sh
python main.py --model [model_name] [--woDiv] [--woEMO] [--woCOG] [--cuda]
```

where model_name could be one of the following: **trs | multi-trs | moel | mime | empdg | cem**. In addition, the extra flags can be used for ablation studies.

## Testing

For reproducibility, download the trained [checkpoint](https://drive.google.com/file/d/1p_Qj5hBQE7e8ailIb5LbZu7NABmeet4k/view?usp=sharing),  put it in a folder named  `saved` and run the following:

```sh
python main.py --model cem --test --model_path save/CEM_19999_41.8034 [--cuda]
```

### Evaluation

Create a folder `results` and move the obtained results.txt for each model to this folder. Rename the files to the name of the model and run the following:

```sh
python src/scripts/evaluate.py 
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
