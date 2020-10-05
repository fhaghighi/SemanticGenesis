# Semantic Genesis - Official Keras Implementation

We provide the official <b>Keras</b> implementation of training Semantic Genesis from scratch on unlabeled images as well as the usage of the pre-trained Semantic Genesis reported in the following paper:

<b>Learning Semantics-enriched Representation via Self-discovery, Self-classification, and Self-restoration</b> <br/>
[Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>, [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Zongwei Zhou](https://github.com/MrGiovanni)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
Arizona State University<sup>1</sup>, </sup>Mayo Clinic <sup>2</sup><br/>
International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI](https://www.miccai2020.org/)), 2020 <br/>

[paper](https://arxiv.org/pdf/2007.06959.pdf) | [code](https://github.com/fhaghighi/SemanticGenesis/) | talk ([YouTube](https://www.youtube.com/embed/II4VkS9Lkdo)) 

## Requirements

+ Linux
+ Python 3.7.5
+ Keras 2.2.4+
+ TensorFlow 1.14.0+

## Training Semantic Genesis on your own unlabeled data

### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/fhaghighi/SemanticGenesis.git
$ cd SemanticGenesis/
$ pip install -r requirements.txt
```

### 2. Preparing data

#### For your convenience, we have provided our own self-discoverd 3D anatomical patterns from LUNA16 dataset as well as their pseudo labels.
Download the data from [Google Drive]() and put it in the `self_discovery` directory. We have provided the training and validation samples for C=50 classes of anatomical patterns. In our study, we trained Semantic Genesis using C=44 classes. For each sample of anatomical pattern, we have extracted 3 multi-resolution cubes from each patient, where each of the three resolutions are saved in files named as 'train_dataN_vwGen_ex_ref_fold1.0.npy',  *N*=1,2,3. For each 'train_dataN_vwGen_ex_ref_fold1.0.npy' file, there is a corresponding 'train_labelN_vwGen_ex_ref_fold1.0.npy' file, which contains the pseudo labels of the discovered anatomical patterns.  


- The processed anatomical patterns directory structure
```
Semantic_Genesis_data/
    |--  train_data1_vwGen_ex_ref_fold1.0.npy  : training data - resolution 1
    |--  train_data2_vwGen_ex_ref_fold1.0.npy  : training data - resolution 2
    |--  train_data3_vwGen_ex_ref_fold1.0.npy  : training data - resolution 3
    |--  val_data1_vwGen_ex_ref_fold1.0.npy    : validation data
    |--  train_label1_vwGen_ex_ref_fold1.0.npy : training labels - resolution 1
    |--  train_label2_vwGen_ex_ref_fold1.0.npy : training labels - resolution 2
    |--  train_label3_vwGen_ex_ref_fold1.0.npy : training labels - resolution 3
    |--  val_label1_vwGen_ex_ref_fold1.0.npy   : validation labels
   
```

####  You can perform the self-discovery on your own dataset following the steps below:

**Step 1**: Divide your training data into the train and validation folders, and put them in the `dataset` directory. 

**Step 2**: Train an auto-encoder using your data. The pre-trained model will be saved into `self_discovery/Checkpoints/Autoencoder/` directory.  

```bash
python -W ignore self_discovery/train_autoencoder.py 
--data_dir dataset/ 
```
**Step 3**: Extract and save the deep features of each patient in the dataset using the pre-trained auto-encoder:

```bash
python -W ignore self_discovery/feature_extractor.py 
--data_dir dataset/  
--weights self_discovery/Checkpoints/Autoencoder/Vnet_autoencoder.h5
```

**Step 4**: Extract 3D anatomical patterns from train and validation images. The data and their labels will be save into `self_discovery/Semantic_Genesis_data` directory.

```bash
python -W ignore self_discovery/pattern_generator_3D.py 
--data_dir dataset/  
--multi_res

```

### 3. Pre-train Semantic Genesis 
```bash
python -W ignore keras/train.py
--data_dir self_discovery/Semantic_Genesis_data
```
Your pre-trained Semantic Genesis will be saved at `./keras/Checkpoints/semantic_genesis_chest_ct.h5`.

## Citation
If you use our source code and/or refer to the baseline results published in the paper, please cite our [paper](https://arxiv.org/pdf/2007.06959.pdf) by using the following BibTex entry:

```
@InProceedings{haghighi2020learning,
author="Haghighi, Fatemeh and Hosseinzadeh Taher, Mohammad Reza and Zhou, Zongwei and Gotway, Michael B. and Liang, Jianming",
title="Learning Semantics-Enriched Representation via Self-discovery, Self-classification, and Self-restoration",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="137--147",
isbn="978-3-030-59710-8",
url="https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_14"
}
```

## Acknowledgement
We thank [Fatemeh Haghighi](https://github.com/fhaghighi) and [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher) for their implementation of Semantic Genesis in Keras. Credit to [Models Genesis](https://github.com/MrGiovanni/ModelsGenesis) by [Zongwei Zhou](https://github.com/MrGiovanni). We build 3D U-Net architecture by referring to the released code at [ellisdg/3DUnetCNN](https://github.com/ellisdg/3DUnetCNN). This is a patent-pending technology.
