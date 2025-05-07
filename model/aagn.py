# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes * 128, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes * 128, in_planes * 8, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes * 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        #self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch // 2, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch // 2, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, y):
        B, C, H, W = x.shape
        #h = self.group_norm(x)
        q = self.proj_q(y)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C // 2)

        k = k.view(B, C // 2, H * W)
        w = torch.bmm(q, k) * (int(C // 2) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class AdaptiveAffinityGraph(nn.Module):

    def __init__(self, in_dim):
        super(AdaptiveAffinityGraph, self).__init__()

        self.in_dim = in_dim
        self.encode_torso = nn.Sequential(nn.Conv2d(7, in_dim, kernel_size=3, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU())

        self.encode_arm = nn.Sequential(nn.Conv2d(7, in_dim, kernel_size=3, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU())

        self.encode_leg = nn.Sequential(nn.Conv2d(7, in_dim, kernel_size=3, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(),
                            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride = 2, padding=1),
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU())

        self.atten_arm_self = AttnBlock(in_dim)
        self.atten_arm_leg = AttnBlock(in_dim)
        self.atten_arm_torso = AttnBlock(in_dim)
        self.atten_leg_self = AttnBlock(in_dim)
        self.atten_leg_arm = AttnBlock(in_dim)
        self.atten_leg_torso = AttnBlock(in_dim)
        self.atten_torso_self = AttnBlock(in_dim)
        self.atten_torso_arm = AttnBlock(in_dim)
        self.atten_torso_leg = AttnBlock(in_dim)

        self.squeeze_1 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_2 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_3 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_4 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_5 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_6 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_7 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_8 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.squeeze_9 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)

        self.channel_atten = ChannelAttention(9)

        self.sigmoid = nn.Sigmoid()

    def structure_encoder(self, origin_arm_mask, origin_leg_mask, origin_torso_mask):

        feature_arm = self.encode_arm(origin_arm_mask)
        feature_torso = self.encode_torso(origin_torso_mask)
        feature_leg = self.encode_leg(origin_leg_mask)

        return feature_arm, feature_torso, feature_leg

    def forward(self, origin_input, PAF_mag):
        """
            inputs :
                x : input feature maps( B x C x H x W)
                Y : ( B x C x H x W), 1 denotes connectivity, 0 denotes non-connectivity
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, height, width = origin_input.size()

        # PAF_mag = PAF_mag.contiguous()
        arm_mask = PAF_mag[:, 0:4, : ,: ]
        leg_mask = PAF_mag[:, 4:8, : ,: ]
        torso_mask = PAF_mag[:, 8:12, : ,: ]

        origin_arm_mask = torch.cat((arm_mask, origin_input), dim = 1)
        origin_leg_mask = torch.cat((leg_mask, origin_input), dim = 1)
        origin_torso_mask = torch.cat((torso_mask, origin_input), dim = 1)

        feature_arm, feature_torso, feature_leg = self.structure_encoder(origin_arm_mask, origin_leg_mask, origin_torso_mask)

        arm_self_atten_map = self.atten_arm_self(feature_arm, feature_arm)
        arm_leg_atten_map = self.atten_arm_leg(feature_arm, feature_leg)
        arm_torso_atten_map = self.atten_arm_leg(feature_arm, feature_torso)

        leg_self_atten_map = self.atten_arm_self(feature_leg, feature_leg)
        leg_arm_atten_map = self.atten_arm_leg(feature_leg, feature_arm)
        leg_torso_atten_map = self.atten_arm_leg(feature_leg, feature_torso)

        torso_self_atten_map = self.atten_arm_self(feature_torso, feature_torso)
        torso_leg_atten_map = self.atten_arm_leg(feature_torso, feature_leg)
        torso_arm_atten_map = self.atten_arm_leg(feature_torso, feature_arm)

        squeeze_arm_self_atten_map = self.squeeze_1(arm_self_atten_map)
        squeeze_arm_leg_atten_map = self.squeeze_2(arm_leg_atten_map)
        squeeze_arm_torso_atten_map = self.squeeze_3(arm_torso_atten_map)

        squeeze_leg_self_atten_map = self.squeeze_1(leg_self_atten_map)
        squeeze_leg_arm_atten_map = self.squeeze_2(leg_arm_atten_map)
        squeeze_leg_torso_atten_map = self.squeeze_3(leg_torso_atten_map)

        squeeze_torso_self_atten_map = self.squeeze_1(torso_self_atten_map)
        squeeze_torso_leg_atten_map = self.squeeze_2(torso_leg_atten_map)
        squeeze_torso_arm_atten_map = self.squeeze_3(torso_arm_atten_map)

        channel_atten_input = torch.cat((squeeze_arm_self_atten_map, squeeze_arm_leg_atten_map, squeeze_arm_torso_atten_map, squeeze_leg_self_atten_map, \
                                         squeeze_leg_arm_atten_map, squeeze_leg_torso_atten_map, squeeze_torso_self_atten_map, squeeze_torso_leg_atten_map, \
                                            squeeze_torso_arm_atten_map), dim = 1)

        channel_atten_output = self.channel_atten(channel_atten_input)

        return torch.cat((channel_atten_output[:, 0, :, :].unsqueeze(1) * arm_self_atten_map , channel_atten_output[:, 1, :, :].unsqueeze(1) * arm_leg_atten_map , channel_atten_output[:, 2, :, :].unsqueeze(1) * arm_torso_atten_map \
        , channel_atten_output[:, 3, :, :].unsqueeze(1) * leg_self_atten_map , channel_atten_output[:, 4, :, :].unsqueeze(1) * leg_arm_atten_map , channel_atten_output[:, 5, :, :].unsqueeze(1) * leg_torso_atten_map , \
        channel_atten_output[:, 6, :, :].unsqueeze(1) * torso_self_atten_map , channel_atten_output[:, 7, :, :].unsqueeze(1) * torso_leg_atten_map , channel_atten_output[:, 8, :, :].unsqueeze(1) * torso_arm_atten_map ), dim = 1)


class FlowGenerator(nn.Module):
    def __init__(self, n_channels, deep_supervision=False):
        super(FlowGenerator, self).__init__()
        self.deep_supervision = deep_supervision

        self.BPAA = AdaptiveAffinityGraph(in_dim=6)
  
        ####################################
        self.head = conv_layer(n_channels, 64)
        self.downlayer1 = conv_layer(64, 64)
        self.downsample1 = nn.MaxPool2d(2) #128

        self.downlayer2 = conv_layer(64, 128)
        self.downlayer3 = conv_layer(128, 128)
        self.downsample2 = nn.MaxPool2d(2) #64

        self.downlayer4 = conv_layer(128, 256)
        self.downlayer5 = conv_layer(256, 256)
        self.downsample3 = nn.MaxPool2d(2)

        self.downlayer6 = conv_layer(256, 512)
        self.downlayer7 = conv_layer(512, 512)
        self.downsample4 = nn.MaxPool2d(2)

        self.extractfea1 = conv_layer(512, 1024)
        self.extractfea2 = conv_layer(1024, 1024)

        self.middle1 = conv_layer(1024, 1024)
        self.middle2 = conv_layer(1024, 1024)

        ############################

        self.uplayer1 = conv_layer(1024, 512)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.uplayer2 = conv_layer(1024, 512)
        self.uplayer3 = conv_layer(512, 256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.uplayer4 = conv_layer(512, 256)
        self.uplayer5 = conv_layer(256, 128)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.uplayer6 = conv_layer(256, 128)
        self.uplayer7 = conv_layer(128, 64)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.outlayer1 = conv_layer(64, 32)
        self.outlayer2 = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.outactivate = nn.Tanh()
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),


        dilation_ksize = 13
        self.dilation= torch.nn.MaxPool2d(kernel_size=dilation_ksize, stride=1, padding=int((dilation_ksize - 1) / 2))

        #####################################
        self.BPAA_encode_head = conv_layer(54, 128)

        self.BPAA_encoder1 = nn.Sequential(
                        conv_layer(128, 256),
                        conv_layer(256, 256),
                        nn.MaxPool2d(2))

        self.BPAA_encoder2 = nn.Sequential(
                        conv_layer(256, 512),
                        conv_layer(512, 512),
                        nn.MaxPool2d(2))

        self.BPAA_encoder3 = nn.Sequential(
                        conv_layer(512, 1024),
                        conv_layer(1024, 1024))
        ###########################################

    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=0.2):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1

        '''
        grid[0,:,:,0] =
        -1, .....1
        -1, .....1
        -1, .....1

        grid[0,:,:,1] =
        -1,  -1, -1
         ;        ;
         1,   1,  1


        image  -1 ~1       -128~128 pixel
        flow   -0.4~0.4     -51.2~51.2 pixel
        '''

        if torch.cuda.is_available():
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        else:
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x


    def forward(self, img, skeleton_map, coef=0.2):
        '''
        img  -1 ~ 1
        skeleton_map  -1 ~ 1
        '''
        PAF_mag = self.dilation(skeleton_map)
        img_concat = torch.cat((img, skeleton_map), dim=1)
        bpaa_in = self.BPAA(img, PAF_mag)

        h_out = self.head(img_concat)
        dl_1 = self.downlayer1(h_out)
        ds_1 = self.downsample1(dl_1)

        dl_2 = self.downlayer2(ds_1)
        dl_3 = self.downlayer3(dl_2)
        ds_2 = self.downsample2(dl_3) 

        bp_eh = self.BPAA_encode_head(bpaa_in)

        dl_4 = self.downlayer4(bp_eh + ds_2)
        dl_5 = self.downlayer5(dl_4)
        ds_3 = self.downsample3(dl_5)

        bp_e1 = self.BPAA_encoder1(bp_eh)

        dl_6 = self.downlayer6(bp_e1 + ds_3)
        dl_7 = self.downlayer7(dl_6)
        ds_4 = self.downsample4(dl_7)

        bp_e2 = self.BPAA_encoder2(bp_e1)
        
        ef_1 = self.extractfea1(bp_e2 + ds_4)
        ef_2 = self.extractfea2(ef_1)

        bp_e3 = self.BPAA_encoder3(bp_e2)

        md_1 = self.middle1(bp_e3 + ef_2)
        md_2 = self.middle2(md_1)

        ul_1 = self.uplayer1(md_2)
        us_1 = self.upsample1(ul_1)

        ul_2 = self.uplayer2(torch.cat((us_1, dl_7), dim = 1))
        ul_3 = self.uplayer3(ul_2)
        us_2 = self.upsample2(ul_3)

        ul_4 = self.uplayer4(torch.cat((us_2, dl_5), dim = 1))
        ul_5 = self.uplayer5(ul_4)
        us_3 = self.upsample3(ul_5)

        ul_6 = self.uplayer6(torch.cat((us_3, dl_3), dim = 1))
        ul_7 = self.uplayer7(ul_6)
        us_4 = self.upsample4(ul_7)

        ol_1 = self.outlayer1(us_4)
        ol_2 = self.outlayer2(ol_1)
        flow = self.outactivate(ol_2)

        flow = flow.permute(0, 2, 3, 1) 

        return  flow
    
