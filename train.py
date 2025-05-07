import torch
from load_data import Flowdataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.aagn import FlowGenerator
from model.gan_components import Discriminator
from tqdm import tqdm
from model.loss import GeneratorLoss

def warp(x, flow, mode='bilinear', padding_mode='border', coff=0.2):
    n, c, h, w = x.size()
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    xv = xv.float() / (w - 1) * 2.0 - 1
    yv = yv.float() / (h - 1) * 2.0 - 1

    if torch.cuda.is_available():
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
    else:
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)

    grid_x = grid + 2 * flow * coff
    warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode, align_corners = False)
    return warp_x

if __name__ == '__main__':

    data_path = 'data/collected_data'
    train_list = []

    with open('data/train_split.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        train_list.append(line)


    batch_size = 4
    input_size = 256
    total_epoch = 300

    train_set = Flowdataset(data_path, train_list, input_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 16, drop_last = True)
    
    net_G = FlowGenerator(16)
    net_D = Discriminator()

    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=2e-5, betas=(0.9, 0.999), amsgrad=False)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=2e-5, betas=(0.9, 0.999), amsgrad=False)
    total_iters = 0

    net_G  = net_G.cuda()
    net_D  = net_D.cuda()

    generator_criterion = GeneratorLoss()
    generator_criterion = generator_criterion.cuda()

    for epoch in range(1, total_epoch + 1):
        counts = 0
        total_loss = 0

        for batch_idx, (Preprocessed_img,src_loss_tensor, Preprocessed_img2,flow, skel_map) in tqdm(enumerate(train_loader)):
            
            net_G.train()
            net_D.train()

            Preprocessed_img = Preprocessed_img.cuda()
            src_loss_tensor = src_loss_tensor.cuda()
            Preprocessed_img2 = Preprocessed_img2.cuda()
            flow = flow.cuda()
            skel_map = skel_map.cuda()

            preflow = net_G(Preprocessed_img, skel_map)
            warped_img = warp(src_loss_tensor,preflow)

            optimizer_D.zero_grad()
               
            real_out = net_D(Preprocessed_img2).mean()
            fake_out = net_D(warped_img).mean()

            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            fake_out = net_D(warped_img).mean()
            optimizer_G.zero_grad()
            g_loss = generator_criterion(fake_out, warped_img, Preprocessed_img2, preflow, flow)
            g_loss.backward()
            optimizer_G.step()

            total_loss += g_loss.item()
            counts += 1

            if counts % 50 == 0:
                print(f"iteration:{counts}      average_loss:{total_loss/counts}")
        print('{}_Epoch_End : {}'.format(epoch,total_loss/counts))

        if epoch % 50 == 0:
            torch.save(net_G.state_dict(), 'ckpts/epoch_{}_AAGN.pth'.format(epoch))