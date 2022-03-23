# BACKGROUND IMAGE AVERAGING CELL
# Import packages
import numpy as np
from skimage import io
from os import path
import scipy.ndimage as ndimage
import tifffile

def avg_bkg(config):
    # path to background images
    input_folder = config['image']['base_path2']
    # for generating bkgd images
    bkg_num_images = 10
    input_img_names = [config['image']['bkgimage_prefix']+str(n)+\
                       '.tif' for n in range(2,(bkg_num_images+1))]
    # path for background image output
    output_folder = config['image']['base_path2']
    output_img_name = config['image']['avg_bkgimage']+'.tif'

    # Averages across multiple images (works for single images or stacks)
    # Eg. to make background images
    # Eg. to average foreground images
    # Select settings
    # Sigma is the sigma for the gaussian filter
    # Use tuple for image stacks:
    #     A 2048x2048 RYGBU imports as (2048, 2048, 5)
    #     Use eg. sigma = (50,50,0) to filter in X and Y in each image, but not across channels
    # For single images, use eg. sigma = (50,50) to filter in X and Y
    sigma = (10,10,0)

    # Create list to store image arrays in
    images = []

    # Import each image and apply gaussian filter
    # Add the smoothed image to the list
    for img in input_img_names:
        name = path.join(input_folder,img)
        img_ = io.imread(name)#[:4,:,:]
        img_ = ndimage.gaussian_filter(img_, sigma=sigma, order=0)
        images.append(img_)
        print(img_.shape)

    # Stack images from list to ndarray
    img_stack = np.stack(images)
    print(img_stack.shape)

    # Average the image values across the images
    img_ave = img_stack.mean(axis=0)
    print(img_ave.shape)

    # Convert the ndarray data to a convenient output data type and shape
    img16 = img_ave.astype('uint16')
    if img16.shape[1] > img16.shape[2]:
        img16trans = np.transpose(img16, (2,0,1))
        print('img16trans shape is: '+str(img16trans.shape))
    else:
        img16trans = img16

    print(img16.shape)

    # Output the new image to the specified location
    new_name = path.join(output_folder,output_img_name)
    if len(img16.shape) == 3:
        tifffile.imsave(new_name, img16trans, photometric='minisblack')
    else:
        tifffile.imsave(new_name, img16)
