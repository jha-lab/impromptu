import os.path
import sys

import argparse
import math
import torch
import torchvision.utils as vutils
from torchvision import datasets, transforms, utils
from datetime import datetime
from torch.optim import Adam
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../utils/')
from create_dataset import *
sys.path.append('../learners/')
from patch_network import PatchNetwork 

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=4)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--dataset', type=str, default='shapes3d')
parser.add_argument('--checkpoint_path',type=str,default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='./logs_dir/')
parser.add_argument('--data_path', default='./datasets/shapes3d_analogy.h5')

parser.add_argument('--lr_main', type=float, default=1e-4)
parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_cosine_anneal_steps', type=int, default=30000)

parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--num_enc_blocks', type=int, default=2)
parser.add_argument('--num_enc_heads', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=128)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps_features', type=int, default=30000)


parser.add_argument('--hard', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
filename = os.path.basename(args.data_path)
log_dir = os.path.join(args.log_path,  args.dataset, filename[:-3] +'_patch_nw_'+str(args.seed))
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
        ]
)

# assert args.dataset in ['dsprites','shapes3d','sprites','clevr']

assert args.dataset in ['shapes3d','clevr','bitmoji']
# if args.dataset == 'dsprites':

#     train_dataset = Dsprites(root=args.data_path, phase='train',transform=transform)
#     val_dataset = Dsprites(root=args.data_path, phase='val',transform=transform)

if args.dataset == 'shapes3d':

    train_dataset = Shapes3D(root=args.data_path, phase='train',transform=transform)
    val_dataset = Shapes3D(root=args.data_path, phase='val',transform=transform)

# elif args.dataset == 'sprites':
    
#     train_dataset = Sprites(root=args.data_path, phase='train',transform=transform)
#     val_dataset = Sprites(root=args.data_path, phase='val',transform=transform)

elif args.dataset == 'clevr':

    train_dataset = CLEVr(root=args.data_path, phase='train',transform=transform)
    val_dataset = CLEVr(root=args.data_path, phase='val',transform=transform)

elif args.dataset == 'bitmoji':

    train_dataset = BitMoji(root=args.data_path, phase='train',transform=transform)
    val_dataset = BitMoji(root=args.data_path, phase='val',transform=transform)


else:

    NotImplementedError
    
train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)


log_interval = train_epoch_size // 5
model = PatchNetwork(args)

if os.path.isfile(args.checkpoint_path):

    print('Loading from checkpoint')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    stagnation_counter = checkpoint['stagnation_counter']
    lr_decay_factor = checkpoint['lr_decay_factor']
    model.load_state_dict(checkpoint['model'],strict= False)


else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    stagnation_counter = 0
    lr_decay_factor = 1.0

model = model.cuda()

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'dvae' not in x[0]), 'lr': args.lr_main},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


def augment_analogy(analogies):
    B, num_examples, _, C, H, W = analogies.shape
    
    inds_perm = torch.randperm(num_examples)
    
    return analogies[:,inds_perm]

def linear_warmup_with_cosine_anneal(step, start_value, peak_value, start_step, peak_step, anneal_step):
    
    assert start_value <= peak_value
    assert start_step <= peak_step
    
    if step < start_step:
        value = start_value
    elif step >= start_step and step <= peak_step:
        a = peak_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (peak_step - start_step)
        value = a * progress + b
    
    elif step >= peak_step and step <= peak_step+anneal_step:
        a = 0.5 * (peak_value - 0.1*peak_value)
        b = 0.5 * (peak_value + 0.1*peak_value)
        progress = (step - peak_step) / anneal_step
        value = a * math.cos(math.pi * progress) + b

    else:
        value = 0.1*peak_value

    return value

def linear_warmup(step, start_value, peak_value, start_step, peak_step):
    
    assert start_value <= peak_value
    assert start_step <= peak_step
    
    if step < start_step:
        value = start_value

    elif step >= start_step and step <= peak_step:
        a = peak_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (peak_step - start_step)
        value = a * progress + b

    else:
        value = peak_value

    return value



def cosine_anneal(step, start_value, final_value, start_step, final_step):
    
    assert start_value >= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    
    return value



def visualize_generation(images, gen, N=8):

    B, num_example, _, C, H, W = images.shape
    images_N = images[:N,].clone()
    gen = gen[:N]
    gt = images_N[:,-1,1].unsqueeze(1).clone()
    images_N[:,-1,1] = gen.reshape(N, C, H, W)
    images_N = images_N[:,-2:].reshape(N,-1, C, H, W)
    vis  = torch.cat((images_N,gt),1)
    return vis.reshape(-1, 3, H, W)


for epoch in range(start_epoch, args.epochs):
    
    model.train()
    
    for batch, images in enumerate(train_loader):

        global_step = epoch * train_epoch_size + batch

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps_features)

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps,
           )
        
        optimizer.param_groups[0]['lr'] = lr_decay_factor*args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor*lr_warmup_factor*args.lr_main

        images = images.cuda()
        B, _, _, C, H, W = images.shape
        images = augment_analogy(images)
        recon, ce, mse = model(images[:,:-1],images[:,-1], tau, args.hard)
        optimizer.zero_grad()
        loss = mse + ce 
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', ce.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                writer.add_scalar('TRAIN/tau_features', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_main', optimizer.param_groups[1]['lr'], global_step)


    support = images[:,:-1]
    query = images[:,-1,0]

    with torch.no_grad():

        gen = model.generate(support, query)
        vis_recon = visualize_generation(images, gen)
        grid = vutils.make_grid(vis_recon, nrow=5, pad_value=0.2)#A:B::C:D
        writer.add_image('TRAIN_relation/epoch={:03}'.format(epoch+1), grid)

 
    with torch.no_grad():

        model.eval()
        val_cross_entropy = 0.
        val_mse = 0.
        
        for batch, images in enumerate(val_loader):
            images = images.cuda()
            B, num_relations, num_examples ,C ,H ,W = images.shape
            recon, ce, mse = model(images[:,:-1],images[:,-1],tau,args.hard)
            val_cross_entropy += ce.item()
            val_mse += mse.item()

        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        val_loss =  val_cross_entropy + val_mse
        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch + 1)

        print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F}'.format(
                epoch+1, val_loss, val_mse))

        if val_loss < best_val_loss:
            stagnation_counter = 0
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if epoch >= 20:

                support = images[:,:-1]
                query = images[:,-1,0]

                gen = model.generate(support, query)
                vis_recon = visualize_generation(images, gen)
                grid = vutils.make_grid(vis_recon, nrow=5, pad_value=0.2)#A:B::C:D
                writer.add_image('VAL_relation/epoch={:03}'.format(epoch+1), grid)

        else:
            stagnation_counter += 1
            if stagnation_counter >= args.patience:
                lr_decay_factor = lr_decay_factor / 2.0
                stagnation_counter = 0

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stagnation_counter': stagnation_counter,
            'lr_decay_factor': lr_decay_factor,
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
