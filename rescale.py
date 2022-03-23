from skimage import io
import yaml
import sys, os, re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_max_drop(channel, q=0.99969):
	'''
	Find highest-intensity droplet in channel.

	Inputs:
	:channel: (NxN ndarray) pixel values from .tif file
	:q: (float) Quantile valuebetween 0 and 1.
				0 = marks everything; 1 = marks nothing

	Outputs:
	:m: (float) Threshold pixel value
	'''

	ch = channel.flatten()
	m = np.quantile(ch, q)

	return m

def mark_max_drop(channel, q):
	'''
	Mark the selected quantile of pixels to check that
	they're in a droplet and not an artifact.
	Outputs intended for plotting.

	Inputs:
	:channel: (NxN ndarray) pixel values from .tif file
	:q: (float) Quantile value passed to get_max_drop.

	Outputs:
	:m: (float) get_max_drop output
	'''

	m = get_max_drop(channel, q)

	return m


def maxdrop_plots(rgb, q=[0.99969]*3, save=False):
	'''
	Plotting function for three-channel images.
	Default quantile value is approximately one droplet.

	Inputs:
	:rgb: (list) 3 NxN ndarrays corresponding to each channel
	:q: (list) List of quantile values to use for each channel
	:save: (str) directory to save image to. if False, does not save.

	Outputs: None, plots images.
	'''

	if len(rgb)!=3:
		raise ValueError("Argument `rgb` must be a list of 3 numpy arrays")

	fig, axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(30, 7))
	ax = axes.ravel()

	maxes = []

	for i in range(3):
		m = mark_max_drop(rgb[i], q[i])
		maxes.append(m)
		masked_array = np.ma.masked_where(rgb[i] >= m, rgb[i])
		cmap = plt.cm.gray  # Can be any colormap that you want after the cm
		cmap.set_bad(color='red')
		ax[i].imshow(masked_array, cmap=cmap)
		ax[i].set_title('Channel '+str(i+1)+'\n'+str(100*q[i])+'th quantile')

	plt.show()

	if save:
		plt.savefig(save)

	return maxes

def rescale_cfg(cfg, maxes, rescale_info=['Reporter']):
	'''
	Updates config dictionary with rescale info and writes it to a new file.

	Inputs:
	:cfg: (dict) Old config dictionary
	:maxes: (list, length 3) Maximum value in each channel, from maxdrop_plots
	:cfg_dir: (str) Directory to write new config file to. Must end in .yml.
	:rescale_info: (list) Relevant additional info for rescale text.
						Loosely intended to provide info about any additional
						elements of the rescale vector.

	Modifies:
	:cfg: (dict) Same as before, but with 'rescale' and 'rescale text' added
				to it. Note that it is *not* a copy of the original.

	Writes new config file to :cfg_dir:
	'''

	mm = max(maxes)

	rescale = [float(round(mm/i, 1)) for i in maxes]
	rescale.append(1)

	cfg['image']['rescale'] = rescale
	cfg['image']['rescale_text'] = ['{:.2f}'.format(x) for x in maxes] + rescale_info

	return cfg

def path_cfg(cfg, img_dir, bg_dir):
	'''
	Creates new config file for one experiment from a template config.

	Inputs:
	:cfg: config_file
	:img_dir: (str) directory, relative to data/raw/, with experiment images
	:bg_dir: (str) directory, relative to data/raw/, with background images

	Outputs:
	:config: (dict) updated config dictionary

	'''
	with open(cfg, 'r') as ymlfile:
	    config = yaml.load(ymlfile)
	# change parameters:
	# path to image files
	config['image']['base_path'] = img_dir

	# path to background images
	config['image']['base_path2'] = bg_dir

	# get list of files in the image directory
	files = os.listdir(config['image']['base_path'])
	# print(files)

	# .tif names
	for tp in ['premerge']:
		# check that at least one image exists in the folder with that suffix
		config['image']['names'][tp] = config['image']['names'][tp]

	return config

def rescale_config(config_file,cfg_NEWname):
	with open(config_file, 'r') as ymlfile:
	    config = yaml.load(ymlfile,Loader=yaml.FullLoader)

	im = io.imread(os.path.join(config['image']['base_path'],
		config['image']['names']['premerge']
		+ '_3_3.tif'))

	rcol, gcol, bcol = im[...,config['image']['dyes'][0]],im[...,config['image']['dyes'][1]],im[...,config['image']['dyes'][2]]

	mx = maxdrop_plots([rcol, gcol, bcol], save=False)

	config = rescale_cfg(config, mx, ['Reporter', '1'])
	print('Rescale vector:',config['image']['rescale'])
	cfg_dir='./'+ cfg_NEWname
	# create new config file
	with open(cfg_dir, 'w') as file:
		yaml.dump(config, file)
