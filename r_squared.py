#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:17:57 2021

@author: joseaceves

Modified by Julie Chen
- rewrote so that bootstrapping is multiprocessed
- changed plotting aesthetic
"""
# import packages
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statistics as st
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import os, time
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from cycler import cycler
## for organizing the abxs/cp_#s
import itertools
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def parameters():
    minorfont = 12
    majorfont = 16
    plt.rcParams['axes.spines.right']=False
    plt.rcParams['axes.spines.top']=False
    plt.rcParams['axes.linewidth']=0.5
    plt.rcParams['axes.labelsize']= majorfont
    plt.rcParams['lines.linewidth']=2
    plt.rcParams['xtick.labelsize']= minorfont
    plt.rcParams['ytick.labelsize']= minorfont
    plt.rcParams['axes.titlesize'] = majorfont
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    # plt uses Type 3 by default
    # not supported by PDF (macOS); switch to 42
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size']= minorfont
    plt.rcParams['xtick.major.width']=1
    plt.rcParams['ytick.major.width']=1
    plt.rcParams['contour.negative_linestyle'] = 'solid'

    plt.rcParams['savefig.bbox']='Tight'
    plt.rcParams['savefig.dpi']=300

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    plt.rcParams['axes.prop_cycle'] = cycler('color', CB_color_cycle)
    return

def conc_df(filename,bug):
    df = pd.read_csv(filename,index_col=0)
    df = df.replace('BLANK_6','BLANK_3')
    df = df.replace('BLANK_5','BLANK_2')
    df = df.replace('BLANK_4','BLANK_1')
    df['bug'] = bug

    # convert conc numbers
    abx_2x = df[df.Label_left.str.contains('CAMHB') & ~df.Label_right.str.contains('CAMHB')]
    abx_2x['abx_name'] = abx_2x['Label_right'].str.split('_').str[0]
    abx_2x['conc'] = abx_2x['Label_right'].str.split('_').str[1]

    info_df = abx_2x[['bug','Label_left','abx_name','conc','t0_normbugs']]

    return info_df

def bug_correlation(bug,output_path,info_df):
    CAMHB_df = info_df[(info_df.Label_left.str.contains('CAMHB'))]
    CAMHB_1_df = CAMHB_df.iloc[::2]
    CAMHB_2_df = CAMHB_df.iloc[1::2]

    # replicates needed for bugs
    med_1_df = CAMHB_1_df.groupby(['abx_name','conc','bug'])['t0_normbugs'].median().reset_index()
    med_2_df = CAMHB_2_df.groupby(['abx_name','conc','bug'])['t0_normbugs'].median().reset_index()

    x = med_1_df['t0_normbugs']
    y = med_2_df['t0_normbugs']

    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2

    plt.figure()
    plt.scatter(x,y)
    plt.title(bug + '_' + str(r_squared))
    plt.savefig(output_path+'per_bug_r2' + '_' + bug +'.png',dpi=300)

    return r_squared

def abx_correlation(bug,output_path,info_df):

    r2_abx = []
    list_abx = []

    for abx in info_df.dropna().abx_name.unique():

        CAMHB_df = info_df[(info_df.Label_left.str.contains('CAMHB'))& (info_df.abx_name==abx)]
        CAMHB_1_df = CAMHB_df.iloc[::2]
        CAMHB_2_df = CAMHB_df.iloc[1::2]

        # replicates needed for bugs
        med_1_df = CAMHB_1_df.groupby(['abx_name','conc','bug'])['t0_normbugs'].median().reset_index()
        med_2_df = CAMHB_2_df.groupby(['abx_name','conc','bug'])['t0_normbugs'].median().reset_index()

        x = med_1_df['t0_normbugs']
        y = med_2_df['t0_normbugs']

        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        r2_abx.append(r_squared)
        list_abx.append(abx)

        plt.figure()
        plt.scatter(x,y)
        plt.title(abx + '_' + bug + '_' + str(r_squared))
        plt.savefig(output_path+'_R2_'+abx +'.png',dpi=300)

        R2_df = pd.DataFrame(data={'R2':r2_abx, 'bug':bug, 'abx':list_abx}).dropna()

    return r2_abx, R2_df

def r2_distr(bug,R2_df):
    fig, ax = plt.subplots(nrows=1,figsize=(8,4))
    ax.hist(R2_df['R2'])
    ax.set_title(bug)
    ax.set_xlabel('R2')
    ax.set_ylabel('counts of 3 conc comparisons')
    ax.set_xlim([0,1])
    ax.set_ylim([0,30])
    return

# calculate r_squared
def R2(x,y):
    '''
    Returns the r-squared value given 2 arrays.
    '''
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    return r_squared

###### bootstrapping analysis ######
from multiprocessing import set_start_method
set_start_method('spawn')

def resample_med(ar, ar_size):
    '''
    Returns the median of resampled values for an array for a given array size.
    '''
    med = np.median(np.random.choice(ar,ar_size,replace=True))
    return med

def bs(ar_size, df1, df2, parallel=False):
    '''
    Returns the R2 value from a single iteration of bootstrapping.
    Resamples for every unique abx/conc given an array size,
    returning the median of this resampling.
    Resamples from both arrays used in the R2 correlation calculation.
    '''
    if parallel:
        bs1 = groupby_for_bs(df1, ar_size)
        bs2 = groupby_for_bs(df2, ar_size)
        r2 = pd.Series(R2(bs1.t0_normbugs, bs2.t0_normbugs), index=['r2'])
        return r2

def bs_parallel(arglist):
    '''
    Enables bootstrapping for functions that require multiple arguments.
    '''
    return bs(*arglist)

def bs_mp(ar_size, df1, df2, max_boot=1001):
    '''
    Parallelizes 1000 resamplings for R2 calculations given an array size,
    and two arrays used for correlation.
    '''
    cores = mp.cpu_count()
    pool = Pool(cores)

    # process the DataFrame by mapping function to each df across the pool
    boot_ = pool.map(bs_parallel, ([ar_size, df1, df2, i] for i in range(1, max_boot)))

    # close down the pool and join
    pool.close()
    pool.join()
    pool.clear()

    booted = pd.concat(boot_, axis=1).reset_index(drop=True)
    av = booted.mean(axis=1)[0]
    err = booted.std(axis=1)[0]

    return av, err

def groupby_for_bs(df, rep):
    '''
    Returns a dataframe whereby each unique bug/abx/conc has its GFP value
    resampled given an array size (sampling depth), returning the median value.
    '''
    med_df = df.groupby(['bug', 'abx_name', 'conc'])['t0_normbugs'].apply(resample_med, ar_size=rep).reset_index()
    return med_df

def bootstrap(output_path,info_df, num_reps=21):
    '''
    Exports a .csv with the average and std of R2 for every sampling depth size,
    whereby the R2 is correlating the values between the two sets of CAMHB/abxs
    and for all unique abx_conc.
    '''
    ## Create the CAMHB_1_df and CAMHB_2_df to compare technical replicates
    CAMHB_df = info_df[(info_df.Label_left.str.contains('CAMHB'))]
    CAMHB_1_df = CAMHB_df.iloc[::2]
    CAMHB_2_df = CAMHB_df.iloc[1::2]

    bs_R2 = pd.DataFrame(columns=['ar_size', 'av', 'err'])

    # Resample/bootstrap 1000 times given a range of array sizes (rep count)
    ## For each resampling, calculate the R2 between the 2-halves for every unique combo
    for k in tqdm(range(1,num_reps)):
        av, err = bs_mp(k, CAMHB_1_df, CAMHB_2_df)
        bs_R2.loc[k] = [k, av, err]

    bs_R2.to_csv(output_path+'bs_R2_to_'+str(num_reps - 1)+'.csv')
    return

def replicate_plot(bug,output_path,df):
    '''
    Plots the average R2 and its std for every sampling depth.
    '''
    plt.figure()
    plt.errorbar(df.ar_size, df.av, yerr=df.err)
    plt.title(bug + ' R2 values based on sampling depth')
    plt.ylabel('mean R2 values (all combos)')
    plt.ylim([0,1])
    plt.xticks(np.arange(0,df.ar_size.max()+1))
    plt.xlabel('microwell sampling depth')
    plt.savefig(output_path+bug+'_replicates_plots.png')
    return
