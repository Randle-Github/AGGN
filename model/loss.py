import torch
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_net = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_net.parameters():
        #     param.requires_grad = False
        # self.loss_net = loss_net
        self.mse_loss = nn.MSELoss()
        self.sm_l1 = nn.SmoothL1Loss()

    def forward(self, out_labels, out_imgs, target_imgs, out_flows, target_flows):
        adversarial_loss = torch.mean(1 - out_labels)
        #perception_loss = self.mse_loss(self.loss_net(out_imgs), self.loss_net(target_imgs))
        imgs_loss = self.sm_l1 (out_imgs, target_imgs)
        flow_loss = self.sm_l1 (out_flows, target_flows)
        return 0.5 * imgs_loss + 0.5 * flow_loss + 0.001 * adversarial_loss #+ 0.006 * perception_loss
    