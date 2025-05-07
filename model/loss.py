import torch
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.sm_l1 = nn.SmoothL1Loss()

    def forward(self, out_labels, out_imgs, target_imgs, out_flows, target_flows):
        adversarial_loss = torch.mean(1 - out_labels)
        imgs_loss = self.sm_l1 (out_imgs, target_imgs)
        flow_loss = self.sm_l1 (out_flows, target_flows)
        return 0.5 * imgs_loss + 0.5 * flow_loss + 0.001 * adversarial_loss
    