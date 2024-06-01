import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage import color
from data.base_dataset import BaseDataset, get_transform
import torch

class VideoColorizationDataset(BaseDataset):
    """This dataset class can load a video, extract frames, and convert frames for colorization."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        parser.add_argument('--video_name', type=str, help='Name of the video file.')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.video_path = os.path.join(opt.dataroot, opt.phase, opt.video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.transform = get_transform(self.opt, convert=False)
        self.frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Ensure the frame is grayscale
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            self.frames.append(gray_frame)
        self.cap.release()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        frame = self.frames[index]
        # print(frame.shape)
        im = Image.fromarray(frame)  # Directly use the gray frame
        # print(im.size)
        im = self.transform(im)
        im = np.array(im)
        im = im.astype(np.float32) / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
        im_t = transforms.ToTensor()(im)
        A = im_t  # Single channel gray image
        B = torch.zeros(2, *A.shape[1:])  # Placeholder for ab channels, which are zero for gray images
        return {'A': A, 'B': B, 'A_paths': self.video_path, 'B_paths': self.video_path}

    def __len__(self):
        """Return the total number of frames in the video."""
        return len(self.frames)
