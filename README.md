# Learning Semantics-enriched Representation via Self-discovery, Self-classification, and Self-restoration

<b>Semantic Genesis</b> is conceptually simple: an encoder-decoder structure with skip connections in between and a classification head at the end of the encoder. The objective for the model is to learn different sets of semantics-enriched representation from multiple perspectives. In doing so, our proposed framework consists of three important components: 
1. Self-discovery of anatomical patterns from similar patients
1. Self-classification of the patterns
1. Self-restoration of the transformed patterns

Specifically, once the self-discovered anatomical pattern set is built, we jointly train the classification and restoration branches together in the model.

\
![Image of framework](https://github.com/fhaghighi/SemanticGenesis/blob/master/images/framework.png)


## Paper
<b>Learning Semantics-enriched Representation via Self-discovery, Self-classification, and Self-restoration</b> <br/>

[Fatemeh Haghighi](https://github.com/mrht27200/test_SG)<sup>1</sup>, [Mohammad Reza Hosseinzadeh Taher](https://github.com/mrht27200/test_SG)<sup>1</sup>,[Zongwei Zhou](https://github.com/MrGiovanni)<sup>1</sup>,[Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1</sup>Arizona State University; </sup>Mayo Clinic; <sup>2</sup><br/>
[MICCAI 2020](https://www.miccai2020.org/), the 23rd International Conference on Medical Image Computing and Computer Assisted Intervention

## Available implementation
<a href="https://keras.io/" target="_blank">
<img alt="Keras" src="https://github.com/fhaghighi/SemanticGenesis/blob/master/images/keras_logo.png" width="200" height="55"> </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pytorch.org/" target="_blank"><img alt="Keras" src="https://github.com/fhaghighi/SemanticGenesis/blob/master/images/pytorch_logo.png" width="200" height="48"></a>  

## Citation
If you use our source code and/or refer to the baseline results published in the paper, please cite our [paper](https://github.com/mrht27200/test_SG/) by using the following BibTex entry:
```
@InProceedings{haghighi2020semantic,
  author="Haghighi, Fatemeh and Hosseinzadeh Taher, Mohammad Reza and Zhou, Zongwei and Gotway, Michael B. and Liang, Jianming",
  title="Learning Semantics-enriched Representation via Self-discovery, Self-classification, and Self-restoration",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020",
  year="2020",
  publisher="Springer International Publishing",
  address="",
  pages="",
  isbn="",
  url="https://link.springer.com/"
}
```

## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the National Institutes of Health (NIH) under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. This is a patent-pending technology.

## License

Released under the [ASU GitHub Project License](https://github.com/mrht27200/test_SG/blob/master/LICENSE) .
