# Saliency Methods
## Introduction
This repository contains code for the following saliency techniques:
* XRAI
* SmoothGrad
* Vanilla Gradients
* Guided Backpropogation
* Integrated Gradients
* Grad-CAM

## Remarks
The methods should work with all models from the [torchvision](https://github.com/pytorch/vision) package. Tested models so far are:
* ResNet variants
* DenseNet variants
* Inception/GoogLeNet

In order for 'Guided Backpropagation' and 'Grad-CAM' to work properly with the 'Inception' and 'GoogLeNet' models, they need to by modified slightly, such that all 'ReLus' are modules of the model rather than function calls. See the [tutorial notebook](https://github.com/hummat/saliency/blob/master/tutorial.ipynb) for details.
    
## Examples
Please have a look at this short [tutorial notebook](https://github.com/hummat/saliency/blob/master/tutorial.ipynb)

## Credits
The implementation follows closely that of the corresponding [TensorFlow saliency](https://github.com/PAIR-code/saliency) repository, reusing its code were applicable (mostly for the XRAI method).

Further inspiration has been taken from [this](https://github.com/utkuozbulak/pytorch-cnn-visualizations) repository.
