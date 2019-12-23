from PIL import Image
import numpy as np
import torch
from vanilla_gradient import VanillaGradient


class GradCam(VanillaGradient):
    def __init__(self, model, conv_layer_index=-1):
        super(GradCam, self).__init__(model)

        self.num_conv_layers = self.count_conv_layers()
        self.conv_layer_index = conv_layer_index
        assert abs(self.num_conv_layers) <= self.num_conv_layers, f"Only {self.num_conv_layers} conv layers present"

        self.conv_output = None
        self.register_hooks()

    def count_conv_layers(self):
        num_conv_layers = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                num_conv_layers += 1
        return num_conv_layers

    def register_hooks(self):
        def save_output(module, input, output):
            self.conv_output = output[0]

        def save_gradient(module, grad_input, grad_output):
            if self.gradient is None:
                self.gradient = grad_output[0]

        conv_layer_index = 0
        modules = list(self.model.modules())
        for i, conv in enumerate(reversed(modules)):
            if isinstance(conv, torch.nn.Conv2d):
                conv_layer_index -= 1
                if conv_layer_index == self.conv_layer_index:
                    for relu in modules[-i:]:
                        if isinstance(relu, torch.nn.ReLU):
                            self.hooks.append(relu.register_forward_hook(save_output))
                            self.hooks.append(relu.register_backward_hook(save_gradient))
                            break
                    if self.gradient is None:
                        self.hooks.append(conv.register_forward_hook(save_output))
                        self.hooks.append(conv.register_backward_hook(save_gradient))
                    break

    def get_mask(self, image_tensor, target_class=None, resize=True):
        super(GradCam, self).get_mask(image_tensor, target_class)

        weights = np.mean(self.gradient.detach().cpu().numpy()[0], axis=(1, 2))
        conv_output = np.moveaxis(self.conv_output.detach().cpu().numpy(), 0, -1)

        cam = np.dot(conv_output, weights)
        cam = np.maximum(cam, 0)
        cam /= cam.max()

        if resize:
            cam = np.array(Image.fromarray(cam).resize((image_tensor.shape[2], image_tensor.shape[3]), Image.ANTIALIAS))
        self.gradient = None
        return cam
