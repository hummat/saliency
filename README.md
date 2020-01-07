# Saliency Methods
## Introduction
This repository contains code for the following saliency techniques:
* XRAI
* SmoothGrad
* Vanilla Gradients
* Guided Backpropogation
* Integrated Gradients
* (Guided) Grad-CAM

## Remarks
The methods should work with all models from the [torchvision](https://github.com/pytorch/vision) package. Tested models so far are:
* VGG variants
* ResNet variants
* DenseNet variants
* Inception/GoogLeNet*

*In order for *Guided Backpropagation* and *Grad-CAM* to work properly with the *Inception* and *GoogLeNet* models, they need to by modified slightly, such that all *ReLUs* are modules of the model rather than function calls.

```python
# This class can be found at the very end of inception.py and googlenet.py respectively.
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)  # Add this line

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)  # Replaces F.relu(x, inplace=True)
```
    
## Examples
For a brief overview on how to use the package, please have a look at this short [tutorial notebook](https://github.com/hummat/saliency/blob/master/tutorial.ipynb). The bare minimum is summarized below.

```python
# Standard imports 
import torchvision

# Import desired utils and methods
from ml_utils import load_image, show_mask
from guided_backprop import GuidedBackprop

# Load model and image
model = torchvision.models.resnet50(pretrained=True)
doberman = load_image('images/doberman.png', size=224)

# Construct a saliency object and compute the saliency mask.
guided_backprop = GuidedBackprop(model)
rgb_mask = guided_backprop.get_mask(image_tensor=doberman)

# Visualize the result
show_mask(rgb_mask, title='Guided Backprop')
```

## Credits
The implementation follows closely that of the corresponding [TensorFlow saliency](https://github.com/PAIR-code/saliency) repository, reusing its code were applicable (mostly for the XRAI method).

Further inspiration has been taken from [this](https://github.com/utkuozbulak/pytorch-cnn-visualizations) repository.
