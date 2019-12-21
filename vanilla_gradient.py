import torch
import numpy as np

from saliency_mask import SaliencyMask


class VanillaGradient(SaliencyMask):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    def get_mask(self, image_tensor, target_class=None):
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits = self.model(image_tensor)
        target = torch.zeros_like(logits)
        target[0][target_class if target_class else logits.topk(1, dim=1)[1]] = 1
        self.model.zero_grad()
        logits.backward(target)
        return np.moveaxis(image_tensor.grad.detach().cpu().numpy()[0], 0, -1)

    def get_smoothed_mask(self, image_tensor, target_class=None, samples=25, std=0.1, process=lambda x: x):
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))

        for sample in range(samples):
            noise = torch.normal(0, std, image_tensor.size())
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]
