import os
from glob import glob

import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_transform_modified
from data.image_folder import make_dataset_with_points


class WithPointsDataset(BaseDataset):
    """A dataset class for paired image dataset, with its facial landmarks .

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset_with_points(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        A_path = glob(os.path.join(AB_path, "A*"))[0]
        B_path = glob(os.path.join(AB_path, "B*"))[0]
        P_path = glob(os.path.join(AB_path, "P*"))[0]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        P = np.load(P_path)

        # apply the same transform to both A and B. its only to_tensor and normalize
        transform = get_transform_modified()

        A = transform(A)
        B = transform(B)

        return {'A': A, 'B': B, 'P': P, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
