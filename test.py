import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import colorcet as cc
import torchvision


from utils import load_image, show_mask, normalize
from vanilla_gradient import VanillaGradient
from guided_backprop import GuidedBackprop
from integrated_gradients import IntegratedGradients


model = torchvision.models.resnet50(pretrained=True)
image = load_image('images/doberman.png', size=224)

guided_backprop = GuidedBackprop(model)
gb_mask = guided_backprop.get_mask(image)
show_mask(gb_mask, cmap=cc.cm.gray)

vg_mask = guided_backprop.get_smoothed_mask(image)
show_mask(vg_mask, cmap=cc.cm.gray)
