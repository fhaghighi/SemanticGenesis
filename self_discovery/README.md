## Requirements

+ Linux
+ Python 3.7.5+
+ Keras 2.2.4+
+ TensorFlow 1.14.0+

## Self-discovery of anatomical patterns from your own unlabeled data

####  You can perform the self-discovery on your own dataset following the steps below:

**Step 1**: Clone the repository and install dependencies
```bash
$ git clone https://github.com/fhaghighi/SemanticGenesis.git
$ cd SemanticGenesis/
$ pip install -r requirements.txt

```

**Step 2**: Divide your training data into the train and validation folders, and put them in the `dataset/` directory. 

**Step 3**: Train an auto-encoder using your data. The pre-trained model will be saved into `self_discovery/Checkpoints/Autoencoder/` directory.  

```bash
python -W ignore self_discovery/train_autoencoder.py 
--data_dir dataset/ 
```
**Step 4**: Extract and save the deep features of each patient in the dataset using the pre-trained auto-encoder:

```bash
python -W ignore self_discovery/feature_extractor.py 
--data_dir dataset/  
--weights self_discovery/Checkpoints/Autoencoder/Unet_autoencoder.h5
```

**Step 5**: Extract 3D anatomical patterns from train and validation images. The data and their labels will be save into `self_discovery/Semantic_Genesis_data` directory.

```bash
python -W ignore self_discovery/pattern_generator_3D.py 
--data_dir dataset/  
--multi_res

```
