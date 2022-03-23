import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kchip.droplets import threshold_image,mask_post_merge
import kchip.io as kchip_io

from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter

from skimage.morphology import remove_small_objects,erosion, disk
from skimage.measure import regionprops,label
import skimage.filters as filters

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

def range_filter(image,disk_size=2):
    ''' Apply range filter to image. This returns the max-min within a given window at each pixel.
    disk_size = 2 original
    Inputs:
        image: m x n, numpy array
    Outputs:
        image_range: range filtered image
    '''

    if config_glo['image']['phase']['disk_sigma']:
        disk_size = config_glo['image']['phase']['disk_sigma']
    else:
        pass
    selem = disk(disk_size)

    image_min = minimum_filter(image,footprint=selem)
    image_max = maximum_filter(image,footprint=selem)

    image_range = image_max-image_min

    return image_range

def rescale(img):
    return (img-img.min()).astype('float')/(img.max()-img.min())

def phase_high_pass(config,phase,phase_bkg_sigma=0.6):
    ''' Sharpen image by computing a high_pass filter. Performing difference of gaussian filtering. Helps with edge detection.
    Inputs:
        - config, the config dictionary
        - phase, the phase image
    Outputs:
        - phase_dog, the high pass filtered image
    '''
    if config['image']['phase']['bkg_sigma']:
        phase_bkg_sigma = config['image']['phase']['bkg_sigma']
    else:
        pass
    phase_bkg = filters.gaussian(phase,sigma=phase_bkg_sigma,preserve_range=True)
    phase_dog = phase-phase_bkg
    phase_dog = (65535*rescale(phase_dog)).astype('uint16')
    return phase_dog

def split_channels(config, image):
    ''' Split the image into dye, and phase channels.
    Inputs:
    - config dictionary
    - 3d np array
    Outputs:
    - dyes, image (sum of dye channels)
    - phase, image
    '''
    dyes = image[:,:,config['image']['dyes']].sum(axis=2)
    phase = image[:,:,config['image']['phase']['phase_idx']]
    return dyes, phase

def phase_mask(config,phase,mask_sigma = 2):
    ''' Compute mask from phase image, to eliminate droplet edges and posts
    Inputs:
    - config, config dictionary
    - phase, the phase image
    Outputs:
    - mask, the phase mask
    '''
    if config['image']['phase']['mask_sigma']:
        mask_sigma = config['image']['phase']['mask_sigma']
    else:
        pass
    return threshold_image(gaussian_filter(phase,mask_sigma))==0

def dye_mask(config,dyes): ### NEED TO FIGURE OUT THE MATH HERE FOR AREA ###
    ''' Segment droplets from dyes image.
    Inputs:
        - dyes, image (1 slice) computed from sum of dye channels
    Outputs:
        -dye_mask_label, the label image of segmented droplets
    '''

    # blur image
    dyes_blurred = gaussian_filter(dyes,3)
    # threshold image
    dyes_mask_all = threshold_image(dyes_blurred)
    # remove small objects
    small_object_size = 100*(config['image']['pixel_size']/(config['image']['pixel_size']/config['image']['objective']*config['image']['bins']))**2

    dyes_mask = remove_small_objects(dyes_mask_all,small_object_size)

    #  Eroding mask to exclude periphery - disk size can be changed
    selem = disk(10);
    dyes_mask_erode = erosion(dyes_mask,selem=selem);

    dyes_mask_label = label(dyes_mask_erode)

    return dyes_mask_label

def median_intensity(regionmask, intensity_image):
    ''' define additional region properties of interest'''
    return np.median(intensity_image)

def sum_intensity(regionmask, intensity_image):
    ''' define additional region properties of interest'''
    return np.sum(intensity_image)

def hi_range_intensity(regionmask, intensity_image):
    ''' define additional region properties of interest'''
    return np.max(intensity_image) - np.min(intensity_image)

def mid_range_intensity(regionmask, intensity_image):
    ''' define additional region properties of interest'''
    return np.percentile(intensity_image,90) - np.percentile(intensity_image,10)

def lo_range_intensity(regionmask, intensity_image):
    ''' define additional region properties of interest'''
    return np.percentile(intensity_image,75) - np.percentile(intensity_image,25)

def initialize_phase(config,timepoint):
    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names'][timepoint])

    phase_df = [0]*len(image_list)
    for xy_idx,xy in enumerate(image_idx):
        print('Now analyzing: '+str(xy[0])+','+str(xy[1]))

        t = kchip_io.read(config,x=xy[0],y=xy[1],t=timepoint,number=4)
        dyes, phase = split_channels(config,t)

        dye_masks = dye_mask(config,dyes)
        phase_masks = phase_mask(config,phase)
        # Create mask from dyes and then from phase: multiply by the binary phase mask to remove the posts.
        mask = dye_masks*phase_masks
        # mask = dye_masks
        # Additional filtering to measure local "roughness"
        phase_signal = range_filter(phase_high_pass(config,phase))

        phase_props = regionprops(mask,phase_signal,extra_properties=(median_intensity,sum_intensity,hi_range_intensity,mid_range_intensity,lo_range_intensity))

        one = np.asarray([p['area'] for p in phase_props])[:,np.newaxis]
        three = np.asarray([p['centroid'] for p in phase_props])
        five = np.asarray([p['mean_intensity'] for p in phase_props])[:,np.newaxis]
        six = np.asarray([p['median_intensity'] for p in phase_props])[:,np.newaxis]
        seven = np.asarray([p['sum_intensity'] for p in phase_props])[:,np.newaxis]
        ten = np.asarray([p['mid_range_intensity'] for p in phase_props])[:,np.newaxis]
        eleven = np.asarray([p['lo_range_intensity'] for p in phase_props])[:,np.newaxis]
        twelve = np.asarray([p['hi_range_intensity'] for p in phase_props])[:,np.newaxis]

        data = np.hstack((one,three,five,six,seven,twelve,ten,eleven))
        phase_df_ = pd.DataFrame(data=data,columns=['Area','ImageY','ImageX',\
                            'Phase_Mean','Phase_Med','Phase_Sum','Phase_HRange','Phase_MRange','Phase_LRange'])
        phase_df_['IndexX']=xy[0]
        phase_df_['IndexY']=xy[1]
        phase_df[xy_idx] = phase_df_

    phase_df = pd.concat(phase_df)
    phase_df.reset_index(inplace=True,drop=True)
    return phase_df

def parallel_phase(xy):
#     print('Now analyzing: '+str(xy[0])+','+str(xy[1]))
    t = kchip_io.read(config_glo,x=xy[0],y=xy[1],t=timepoint_glo,number=4)
    dyes, phase = split_channels(config_glo,t)

    dye_masks = dye_mask(config_glo,dyes)
    phase_masks = phase_mask(config_glo,phase)
    # Create mask from dyes and then from phase: multiply by the binary phase mask to remove the posts.
    mask = dye_masks*phase_masks
    # mask = dye_masks
    # Additional filtering to measure local "roughness"
    phase_signal = range_filter(phase_high_pass(config_glo,phase))

    phase_props = regionprops(mask,phase_signal,extra_properties=\
                  (median_intensity,sum_intensity,hi_range_intensity,mid_range_intensity,lo_range_intensity))

    one = np.asarray([p['area'] for p in phase_props])[:,np.newaxis]
    three = np.asarray([p['centroid'] for p in phase_props])
    five = np.asarray([p['mean_intensity'] for p in phase_props])[:,np.newaxis]
    six = np.asarray([p['median_intensity'] for p in phase_props])[:,np.newaxis]
    seven = np.asarray([p['sum_intensity'] for p in phase_props])[:,np.newaxis]
    ten = np.asarray([p['mid_range_intensity'] for p in phase_props])[:,np.newaxis]
    eleven = np.asarray([p['lo_range_intensity'] for p in phase_props])[:,np.newaxis]
    twelve = np.asarray([p['hi_range_intensity'] for p in phase_props])[:,np.newaxis]

    data = np.hstack((one,three,five,six,seven,twelve,ten,eleven))
    phase_df_ = pd.DataFrame(data=data,columns=['Area','ImageY','ImageX',\
                        'Phase_Mean','Phase_Med','Phase_Sum','Phase_HRange','Phase_MRange','Phase_LRange'])
#     phase_df_['IndexX']=xy[0]
#     phase_df_['IndexY']=xy[1]

    phase_df_.insert(0,'IndexY',xy[1])
    phase_df_.insert(0,'IndexX',xy[0])

    print('Now analyzed: '+str(xy[0])+','+str(xy[1]))
#     print(type(phase_df_),phase_df_.shape)
#     phase_df[xy_idx] = phase_df_

#     phase_df = pd.concat(phase_df)
#     phase_df.reset_index(inplace=True,drop=True)

    return phase_df_

def initialize_phase_parallel(config,timepoint):
    '''Initialize the droplets DataFrame by finding droplets in all images.
    Inputs:
        - config, the config dictionary created from config
    Outputs:
        - droplets (DataFrame), with Image index and position, and RGB data
     '''
    global config_glo
    global len_of_list
    global timepoint_glo
    config_glo = config
    timepoint_glo = timepoint

    # Determine image list
    image_list, image_idx = kchip_io.list_images(config['image']['base_path'],config['image']['names'][timepoint_glo])
    len_of_list = len(image_idx)

    # create the multiprocessing pool
    cores = mp.cpu_count()//int(1)
    pool = Pool(cores)

    # process the dataFrame by mapping function to each df across the pool
    phase_df_list = pool.map(parallel_phase,image_idx)

    # close down the pool and join
    pool.close()
    pool.join()
    pool.clear()

    # Concatenate all the phase_df_list dataframes
    phase_df_all = pd.concat(phase_df_list).reset_index(drop=True)
    return phase_df_all
