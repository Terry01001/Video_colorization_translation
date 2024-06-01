"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, frames_to_video
from util import html
from PIL import Image
import numpy as np

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')
    
    if opt.eval:
        model.eval()


    # first stage test
    if opt.dataset_mode == 'videocolorization':
        frames=[]
        frame_number = 1
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths(frame_number)     # get image paths
            with open(os.path.join(opt.results_dir, opt.name, opt.video_name.split('.')[0], 'fps.txt'), 'w') as f:
                f.write(str(dataset.dataset.fps))
            frame_number += 1
            if i % 500 == 0:  # save images to an HTML file
                print('processing (%05d)-th frame... %s' % (frame_number -1, img_path))
            
            # Save the fake_B result to frames list
            fake_B = visuals['fake_B_rgb']
            fake_B_image = Image.fromarray(fake_B.astype(np.uint8))
            fake_B_image.save(img_path)
            frames.append(np.array(fake_B_image)) 
        
        # Convert frames to video
        video_output_path = os.path.join(opt.results_dir, opt.name, opt.video_name.split('.')[0] + '_colorization.mp4')
        with open(os.path.join(opt.results_dir, opt.name, opt.video_name.split('.')[0], 'fps.txt'), 'r') as f:
            fps = float(f.read().strip())
        frames_to_video(frames, video_output_path, fps)
        print(f'Video saved to {video_output_path}')
        
        # # Convert frames to video
        # video_output_path = os.path.join(opt.results_dir, opt.name, 'output_video.mp4')
        # frames_to_video(frames, video_output_path)
        # print(f'Video saved to {video_output_path}')

    # second stage test (cyclegan)
    elif 'results' in opt.dataroot:
        os.makedirs(os.path.join(opt.results_dir, opt.name, opt.dataroot.split('/')[-1]), exist_ok=True)
        # os.makedirs(os.path.join(opt.results_dir, opt.name, opt.dataroot.split('/')[-1], 'imgs'), exist_ok=True)
        frames = []
        for i, data in enumerate(dataset):
            
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 500 == 0:  # save images to an HTML file
                print('processing (%05d)-th image... %s' % (i + 1, img_path))

            fake = visuals['fake'][0].cpu().numpy().transpose(1,2,0)
            fake_image = Image.fromarray((fake*255).astype(np.uint8))
            #fake_image.save(os.path.join(opt.results_dir, opt.name, opt.dataroot.split('/')[-1]))
            frames.append(np.array(fake_image))

        # convert frames to video
        video_name = opt.dataroot.split('/')[-1] + '_output.mp4'
        video_output_path = os.path.join(opt.results_dir, opt.name, video_name)
        with open(os.path.join(opt.dataroot, 'fps.txt'), 'r') as f:
            fps = float(f.read().strip())
        frames_to_video(frames, video_output_path, fps)
        print(f'Video saved to {video_output_path}')
            

    else:
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

        
        for i, data in enumerate(dataset):
            
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        webpage.save()  # save the HTML

