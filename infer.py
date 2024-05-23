import argparse

import json
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from picai_baseline.unet.training_setup.default_hyperparam import get_default_hyperparams
from picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run
from picai_baseline.unet.training_setup.image_reader import SimpleITKDataset
from monai.transforms import Compose, EnsureType
from picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from scipy.ndimage import gaussian_filter

def create_dataloader(args):
    """
    Creeates a dataloader for the current fold.
    """
    images_dir = Path(args.task_dir) / 'imagesTr'
    image_paths = [images_dir / item for item in os.listdir(images_dir) if item.endswith('.nii.gz')]
    image_paths.sort()

    assert len(image_paths) % 3 == 0, "Number of images not multiple of 3"

    # Group per 3 images
    image_paths = [image_paths[i:i+3] for i in range(0, len(image_paths), 3)]

    # Create dataset and dataloader
    dataset = SimpleITKDataset(image_files=image_paths, transform=Compose([EnsureType()]))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader


def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Inference Script')

    parser.add_argument('--task-dir', type=str, required=True,
                        help="Base path to folder that contains imagesTr")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--weights-dir', type=str, required=True,            
                        help="Path to model checkpoints")
    parser.add_argument('--folds', type=int, nargs='+', required=True, 
                        help="Folds for which to select the weights")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory")
    
    # Model specs
    parser.add_argument('--image_shape', type=int, nargs="+", default=[20, 256, 256],   
                        help="Input image shape (z, y, x)")
    parser.add_argument('--num_channels', type=int, default=3,                
                        help="Number of input channels/sequences")
    parser.add_argument('--num_classes', type=int, default=2,                
                        help="Number of classes at train-time")
    parser.add_argument('--model_type', type=str, default='unet',
                        help="Neural network: architectures")
    parser.add_argument('--model_strides', type=str, default='[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]',
                        help="Neural network: convolutional strides (as string representation)")
    parser.add_argument('--model_features', type=str, default='[32, 64, 128, 256, 512, 1024]',
                        help="Neural network: number of encoder channels (as string representation)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Mini-batch size")
    parser.add_argument('--use_def_model_hp', type=int, default=1,
                        help="Use default set of model-specific hyperparameters")


    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    dataloader = create_dataloader(args)

    args = get_default_hyperparams(args)

    print(args)

    for fold in args.folds:
        print(f"Inferring fold {fold}...")

        model = neural_network_for_run(args=args, device=device)

        weights_file = Path(args.weights_dir) / f'{args.model_type}_F{fold}.pt'
        checkpoint = torch.load(weights_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        model.eval()
        results_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                valid_images = batch[:,0]
                # Create two copies, one flipped, one normal
                valid_images = [valid_images.to(device), torch.flip(valid_images, [4]).to(device)]

                # Make predicitons for flipped and unflipped images
                preds = [
                    torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy()
                    for x in valid_images
                ]

                # Unflip flipped image
                preds[1] = np.flip(preds[1], [3])

                # Average flipped and unflipped image
                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                preds = np.mean([
                    gaussian_filter(x, sigma=1.5)
                    for x in preds
                ], axis=0)

                results_list.append(preds)

        results = np.concatenate(results_list, axis=0)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        np.save(Path(args.output_dir) / f'results_F{fold}.pkl', results)



if __name__ == '__main__':
    main()
