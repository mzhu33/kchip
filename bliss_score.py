#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:48:35 2021

@author: joseaceves
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
import warnings
warnings.filterwarnings('ignore')

## for organizing the abxs/cp_#s
import itertools
import re
from cycler import cycler

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# define plotting parameters
def parameters():
    fontsize = 14
    plt.rcParams['axes.spines.right']=False
    plt.rcParams['axes.spines.top']=False
    plt.rcParams['axes.linewidth']=3
    plt.rcParams['axes.labelsize']=fontsize
    plt.rcParams['lines.linewidth']=2
    plt.rcParams['xtick.labelsize']=fontsize
    plt.rcParams['ytick.labelsize']=fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size']=fontsize
    plt.rcParams['xtick.major.width']=1.5
    plt.rcParams['ytick.major.width']=1.5
    plt.rcParams['contour.negative_linestyle'] = 'solid'

    plt.rcParams['savefig.bbox']='Tight'
    plt.rcParams['savefig.dpi']=300

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    plt.rcParams['axes.prop_cycle'] = cycler('color', CB_color_cycle)

#Establish DataFrame from csv
def establish_df(df_):

    df = df_[['Label_left','Label_right','t0_norm2']]
    #Clear df of blank droplets
    df = df[~(df['Label_left'].str.contains('Blank') | df['Label_right'].str.contains('Blank'))]
    df = df[~(df['Label_left'].str.contains('BLANK') | df['Label_right'].str.contains('BLANK'))]

    #Normalize Value to Bug-Bug Control
    bugbugctrl = df[(df['Label_right'].str.contains("BUGS"))&(df['Label_left'].str.contains("BUGS"))].median()
    aabx = df[~df['Label_left'].str.contains('BUGS|CAMHB') & ~df['Label_right'].str.contains('BUGS|CAMHB')]
    aabx['norm'] = aabx['t0_norm2'] / bugbugctrl.values

    return df,aabx,bugbugctrl

def df_stats(aabx):
    # getting med std and counts for all abx abx pairs
    aabx_med = aabx.groupby(['Label_left','Label_right']).median()
    aabx_std = aabx.groupby(['Label_left','Label_right']).std()
    aabx_count = aabx.groupby(['Label_left','Label_right']).count()

    order = sorted(aabx['Label_left'].unique())

    # putting info in sq matrix form
    abxm_med = pd.DataFrame(index=order, columns=order)
    abxm_std = pd.DataFrame(index=order, columns=order)
    abxm_count = pd.DataFrame(index=order, columns=order)

    for i in aabx_med.index:
        abxm_med.loc[i[0],i[1]] = aabx_med.loc[i,'norm']
        abxm_med.loc[i[1],i[0]] = aabx_med.loc[i,'norm']
        abxm_std.loc[i[0],i[1]] = aabx_std.loc[i,'norm']
        abxm_std.loc[i[1],i[0]] = aabx_std.loc[i,'norm']
        abxm_count.loc[i[0],i[1]] = aabx_count.loc[i,'norm']
        abxm_count.loc[i[1],i[0]] = aabx_count.loc[i,'norm']

    # getting custom annotation df for annotating heatmaps for replicates with <= 3 replicate counts
    # denoting with an asterick
    abxm_annot = pd.DataFrame(index=abxm_count.index,columns=abxm_count.columns)
    for i in abxm_count.index:
        for j in abxm_count.columns:
            if abxm_count.loc[i,j]>=5:
                abxm_annot.loc[i,j]=''
            else:
                abxm_annot.loc[i,j]='*'
    return abxm_med, abxm_std, abxm_count, abxm_annot,order


# creating mask to ignnore upper half triangle on sq matrix plots
def mask(data_frame):
    mask = np.zeros_like(data_frame, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return mask



def base_plots(mask,strain,output_path,abxm_count,abxm_med,abxm_std,abxm_annot):
    # abxm_count plot
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(abxm_count.fillna(0),annot=True,mask=mask,cmap='coolwarm',cbar_kws={'pad':0.01,'aspect':50}, xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(strain+' replicate counts')
    plt.savefig(output_path+'_abxabx_countsmap.png',bbox_inches='tight',dpi=300)

    # growth plot
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(abxm_med.fillna(0),annot=abxm_annot,fmt = '',\
                mask=mask,cmap='coolwarm',cbar_kws={'pad':0.01,'aspect':50}, xticklabels=True, yticklabels=True)
    for i in np.arange(0,abxm_med.shape[1]+1,3):
        plt.axvline(i, color='white', lw=2)
        plt.axhline(i, color='white', lw=2)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(strain+' Median Relative Growth (A.F.U.)')
    plt.savefig(output_path+'_abxabx_heatmap.png',bbox_inches='tight',dpi=300)

    # std plot
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(abxm_std.fillna(0),annot=abxm_annot,fmt = '',\
                mask=mask,cmap='coolwarm',cbar_kws={'pad':0.01,'aspect':50}, xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(strain+' Stdev Relative Growth (A.F.U.)')
    plt.savefig(output_path+'_abxabx_stdevmap.png',bbox_inches='tight',dpi=300)



# since all of my data is just the percent growth I need to change it
# to percent growth inhibition by subtracting all values from 1
# abxm_med is my abx v. abx matrix of all the percent growth of all abx v. abx combos

def growth_inh(abxm_med,df,bugbugctrl):
    growthinh = 1 - abxm_med


    # create med growth inh vals of abx-bug combos
    aabxbug =  df[(df['Label_left'].str.contains('BUGS')) | ((df['Label_right'].str.contains('BUGS')))]
    abx_bug = aabxbug[~(aabxbug['Label_left'].str.contains('BUGS') & (aabxbug['Label_right'].str.contains('BUGS')))]
    abx_bug['norm'] = abx_bug['t0_norm2']/bugbugctrl.values

    abx_bug_left = abx_bug[~abx_bug['Label_left'].str.contains('BUGS|CAMHB')]
    abx_bug_left['abx'] = abx_bug_left['Label_left']
    abx_bug_right = abx_bug[~abx_bug['Label_right'].str.contains('BUGS|CAMHB')]
    abx_bug_right['abx'] = abx_bug_right['Label_right']

    # merged with abx labels
    abx_bug_sort = pd.concat([abx_bug_left, abx_bug_right])
    abx_bug_sort['growthinh'] = 1-abx_bug_sort['norm']

    abx_bug_med = abx_bug_sort.groupby(['abx'])['growthinh'].median().reset_index()
    abx_bug_med['growthinh'][abx_bug_med['growthinh']<0] = 0

    # make the abx-bugs df effect
    abx_sort_ = abx_bug_med.set_index('abx')

    return growthinh,abx_sort_


def bliss(growthinh,abx_sort_,abx_a,abx_b):
    try:
        Eab = growthinh.loc[abx_a,abx_b]
        Ea = abx_sort_.loc[abx_a,'growthinh']
        Eb = abx_sort_.loc[abx_b,'growthinh']
        bliss = Eab - (Ea+Eb-Ea*Eb)
        return bliss
    except KeyError:
        bliss = 0
        return bliss

def df_bliss(order,output_path, growthinh,abx_sort_):
    bliss_df = pd.DataFrame(index=order,columns=order)
    for i in bliss_df.index:
        for j in bliss_df.columns:
            bliss_df.loc[i,j] = bliss(growthinh,abx_sort_,i,j)
    bliss_df.to_csv(output_path+'_abxabx_bliss_all.csv')

    return bliss_df


def indv_bliss(output_path,bliss_df,strain,mask,abxm_med):
    # plot distribution of individual bliss scores
    plt.figure(figsize=(5,3))
    plt.hist(list(bliss_df.values.flatten()),bins=100)
    plt.ylabel('Frequency')
    plt.xlabel('bliss score of all abx-abx combos')
    plt.title(strain+' bliss score distribution')
    plt.savefig(output_path+'_abxabx_blissscore_dist.png',bbox_inches='tight',dpi=300)

    # plot individual bliss scores
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(bliss_df.fillna(0),mask=mask,cmap='coolwarm',cbar_kws={'pad':0.01,'aspect':50},vmin=-bliss_df.abs().max().max(), vmax=bliss_df.abs().max().max(), xticklabels=True, yticklabels=True)
    for i in np.arange(0,abxm_med.shape[1]+1,3):
        plt.axvline(i, color='white', lw=2)
        plt.axhline(i, color='white', lw=2)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(strain+' individual bliss scores')
    plt.savefig(output_path+'_abxabx_bliss_heatmap.png',bbox_inches='tight',dpi=300)


def sum_bliss(output_path,bliss_df,strain,order):
    abx_n = np.unique([i[0:-2] for i in order])
    abx_inputs = len(order)

    # creates bliss_sum df where the axes are just abx_[1-22] (no concentration denoted here since
    # I am summing all possible concentration pairs i.e. 9 different pairs in a 3x3 checkerboard)
    bliss_sum = pd.DataFrame(index=abx_n,columns=abx_n)
    # summing those 3x3 checkerboards and filling in bliss_sum
    # NOTE THAT THIS WOULD BE DIFFERNT FROM ABX-CP PAIRS since only one conc of cp
    for i in enumerate(np.arange(0,abx_inputs,3)):
        for j in enumerate(np.arange(0,abx_inputs,3)):
            summed = np.sum(bliss_df.iloc[i[1]:i[1]+3,j[1]:j[1]+3].values.flatten())
            bliss_sum.iloc[i[0],j[0]] = summed
    bliss_sum.to_csv(output_path+'_abxabx_bliss_summed.csv')

    # plot distribution of summed bliss scores
    plt.figure(figsize=(5,3))
    plt.hist(list(bliss_sum.values.flatten()),bins=100)
    plt.ylabel('Frequency')
    plt.xlabel('bliss score of all abx-abx combos')
    plt.title(strain+' '+' summed bliss score distribution')
    plt.savefig(output_path+'_abxabx_blisssum_dist.png',bbox_inches='tight',dpi=300)

    # creating mask to ignnore upper half triangle on sq matrix plots
    mask1 = np.zeros_like(bliss_sum, dtype=np.bool)
    mask1[np.triu_indices_from(mask1)] = True

    # plot heatmap of summed bliss scores
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(bliss_sum.fillna(0),mask=mask1,cmap='coolwarm',cbar_kws={'pad':0.01,'aspect':50},vmin=-bliss_df.abs().max().max(), vmax=bliss_df.abs().max().max(), xticklabels=True, yticklabels=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(strain+' summed bliss scores')
    plt.savefig(output_path+'_abxabx_bliss_sumheatmap.png',bbox_inches='tight',dpi=300)
