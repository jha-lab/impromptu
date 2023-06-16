import sys
import os
sys.path.append('../utils/')

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from create_dataset import *

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default='shapes3d')
parser.add_argument('--data_path', default='./datasets/shapes3d_analogy.h5')
parser.add_argument('--logs_dir', default='./results/')
parser.add_argument('--phase',default='val')
args = parser.parse_args()

torch.manual_seed(args.seed)
transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
        ]
)

assert args.dataset in ['shapes3d', 'clevr', 'bitmoji']

if args.dataset == 'shapes3d':
    val_dataset = Shapes3D(root=args.data_path, phase = args.phase, transform=transform)

 
elif args.dataset == 'clevr':
    val_dataset = CLEVr(root=args.data_path, phase=args.phase,transform=transform)

elif args.dataset == 'bitmoji':
    val_dataset = BitMoji(root=args.data_path, phase=args.phase, transform=transform)


val_sampler = None
loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
val_epoch_size = len(val_loader)



with torch.no_grad():

    if args.dataset == 'clevr':
        num_examples = 3
    else:
        num_examples = 4


    out_dir = args.logs_dir + f'/pixel'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    val_mse = 0.
    count = 0
    
    for batch, images in enumerate(val_loader):

        support = images[:,:num_examples]
        query = images[:,-1,0]
        gt = images[:,-1,1]
        add = support[:,0,1] - support[:,0,0]
        gen = query + add 
        
        loss = ((gen - gt)**2).sum()/support.shape[0]
        val_mse += loss.item()
        gen = gen.clamp(0,1)
        #save gen images as a png file
        gen = gen.cpu().numpy()
        gen = np.transpose(gen, (0, 2, 3, 1))
        gen = gen * 255
        gen = gen.astype(np.uint8)
        for i in range(gen.shape[0]):
            img = Image.fromarray(gen[i])
            img.save(os.path.join(out_dir, f'{count}.png'))
            count += 1


    val_mse /= (val_epoch_size)
    print(f'Extrapolation MSE ==> {val_mse:.4f}')