import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import imageio
from glob import glob
import torch
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

from module import RaceClassifier

race2class = {
    'African': 0,
    'Asian': 1,
    'Caucasian': 2,
    'Indian': 3
}
class2race = {value: key for key, value in race2class.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='input dir')
    parser.add_argument('--output_dir', type=str, help='dir where file with predicted ethnicities and visualizations will be saved to')
    parser.add_argument('--ckpt', type=str, help='path to a checkpoint')

    parser.add_argument('--visualize', action='store_true', default=False, 
        help='whether to visualize predictions in grids')
    parser.add_argument('--n_vis_samples_per_pic', type=int, default=40, 
        help='# samples per visualized picture')
    parser.add_argument('--n_vis_rows_per_pic', type=int, default=5, 
        help='# rows per visualized picture; must divide n_vis_samples_per_pic')
    parser.add_argument('--figsize_h', type=int, default=20, 
        help='matplotlib figsize height')
    parser.add_argument('--figsize_w', type=int, default=20, 
        help='matplotlib figsize width')

    args = parser.parse_args()

    fns = list(glob(os.path.join(args.input_dir, '**', '*.jpg'), recursive=True)) + \
            list(glob(os.path.join(args.input_dir, '**', '*.jpeg'), recursive=True)) + \
            list(glob(os.path.join(args.input_dir, '**', '*.png'), recursive=True))
    
    print('filenames found:', len(fns))

    classifier = RaceClassifier(num_classes=4)
    classifier = classifier.to('cuda:0')
    model = RaceClassifier.load_from_checkpoint(args.ckpt, 
                                                num_classes=4)
    model.to('cuda:0')
    _ = model.eval()

    n_rows = args.n_vis_rows_per_pic
    os.makedirs(args.output_dir, exist_ok=True)

    all_preds = []
    if args.visualize:    
        n_cols = args.n_vis_samples_per_pic // n_rows
        fig, ax = plt.subplots(n_rows, n_cols, 
                               figsize=(args.figsize_h, args.figsize_w))
        n_pics_visualized = 0
        sample_in_vis_idx = 0

    for i, fn in tqdm(enumerate(fns), total=len(fns)):
        img = imageio.imread(fn)
        h, w = img.shape[:2]
        img = img.astype(np.float32) / 255.
        inp = img * 2 - 1
        inp = torch.tensor(inp)
        inp = inp.to('cuda:0')
        inp = rearrange(inp, 'h w (b c) -> b c h w', b=1)
        
        pred = model(inp)
        pred_label = class2race[torch.argmax(pred).item()]
        all_preds.append((fn, pred_label))

        if args.visualize:
            ax[sample_in_vis_idx // n_cols][sample_in_vis_idx % n_cols].imshow(img)
            ax[sample_in_vis_idx // n_cols][sample_in_vis_idx % n_cols].set_title(f'Pred: {pred_label}')
            sample_in_vis_idx += 1
            
        if (i + 1) % args.n_vis_samples_per_pic == 0:
            plt.savefig(os.path.join(args.output_dir, f'{n_pics_visualized}.jpg'))
            n_pics_visualized += 1
            fig, ax = plt.subplots(n_rows, n_cols, 
                               figsize=(args.figsize_h, args.figsize_w))
            sample_in_vis_idx = 0
    

    with open(os.path.join(args.output_dir, 'pred_ethnicities.txt'), 'w') as fout:
        fout.write('\n'.join([name + ' ' + ethn 
                              for name, ethn in all_preds]))
    