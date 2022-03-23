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
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

def bootstrap_median(ar):
    # resample from all dataframe rows
    return np.median(np.random.choice(ar, len(ar)))

def boot(df,boot_val,parallel=False):
    '''
    boot_val: column name of values to bootstrap sample
    '''
    med_agg = bootstrap_median if parallel else 'median'
    grouped = df[['Label_left','Label_right',boot_val]].groupby(['Label_left','Label_right'])
    return grouped.aggregate(med_agg)

def boot_parallel(arglist):
    return boot(*arglist)

def MIC_boot_(d_area_trimmed,tp,name_media,name_bugs):
    '''
    name_bugs: label with bugs only
    name_media: label with media only
    d_area_trimmed dataframe: cleaned up df from image analysis pipeline, post filtering
    tp: # timepoints
    '''
    bugmedia_drops = d_area_trimmed[(d_area_trimmed.Label_left.str.contains(name_bugs)) & (d_area_trimmed.Label_right.str.contains(name_media))]
    d_area_trimmed['comboID'] = d_area_trimmed[['Label_left','Label_right']].apply(
    lambda x: ','.join(x.dropna().astype(str)),axis=1)
    abx_med_df = d_area_trimmed[d_area_trimmed['comboID'].str.contains(name_media)]
    abx_med_df['new_comboID'] = abx_med_df['comboID'].replace({','+name_media:'',name_media+',':''},regex=True)
    abx_med_df['abx'] = abx_med_df['new_comboID'].str.split('_').str[0]
    abx_med_df['conc'] = abx_med_df['new_comboID'].str.split('_').str[1]
    d_area_trimmed['abx'] = d_area_trimmed.Label_left.apply(lambda x: x.split('_')[0])
    # abxs = abx_med_df.abx.unique()

    MED = d_area_trimmed[['Label_left','Label_right','t0_norm2','abx']].groupby(['Label_left','Label_right']).median()

    for i in range(tp):
        bugmed_med = bugmedia_drops['t'+str(i)+'_norm2'].median()
        d_area_trimmed['t'+str(i)+'_normbugs'] = d_area_trimmed['t'+str(i)+'_norm2'] / bugmed_med
        MED['t'+str(i)+'_normbugs'] = d_area_trimmed['t'+str(i)+'_normbugs'].median()
        d_area_trimmed = d_area_trimmed.dropna(subset=['t'+str(i)+'_normbugs'])

        cores = mp.cpu_count()//3
        pool = Pool(cores)
        boot_ = pool.map(boot_parallel,([d_area_trimmed,'t'+str(i)+'_normbugs',j] for j in range(1,1001)))
        pool.close()
        pool.join()
        pool.clear()

        booted = pd.concat(boot_, axis=1)
        MED['t'+str(i)+'_normbugs_SE'] = np.std(booted.values,axis=1)
        MED = MED.reset_index()
        MED['comboID'] = MED[['Label_left','Label_right']].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1)
        SE_dict = dict(zip(MED.comboID, MED['t'+str(i)+'_normbugs_SE']))
        d_area_trimmed['t'+str(i)+'_normbugs_SE'] = d_area_trimmed['comboID'].map(SE_dict)
    return d_area_trimmed,MED

def z_factor(pos_,neg_,pos_std,neg_std):
    z = 1 - (3*(pos_std + neg_std) / (abs(pos_ - neg_)))
    return z

def calc_zfactor(d_area_trimmed,negctrl_,posctrl_,t):
    '''
    negctrl_: 'CAMHB'
    posctrl_: 'BUGS'
    t = 0 in t0_norm2
    '''
    neg_SE = d_area_trimmed[(d_area_trimmed.Label_left.str.contains(negctrl_)) & (d_area_trimmed.Label_right.str.contains(negctrl_))]['t'+str(t)+'_normbugs_SE'].mean()
    neg_med = d_area_trimmed[(d_area_trimmed.Label_left.str.contains(negctrl_)) & (d_area_trimmed.Label_right.str.contains(negctrl_))]['t'+str(t)+'_normbugs'].median()

    pos_SE = d_area_trimmed[(d_area_trimmed.Label_left.str.contains(posctrl_)) & (d_area_trimmed.Label_right.str.contains(negctrl_))]['t'+str(t)+'_normbugs_SE'].mean()
    pos_med = d_area_trimmed[(d_area_trimmed.Label_left.str.contains(posctrl_)) & (d_area_trimmed.Label_right.str.contains(negctrl_))]['t'+str(t)+'_normbugs'].median()

    z_ = z_factor(pos_med,neg_med,pos_SE,neg_SE)
    print(z_,pos_med,neg_med,pos_SE,neg_SE)
    return z_

def plot_MIC(d_area_trimmed,out_path,t,name_media,name_bugs):
    abx_med_df = d_area_trimmed[d_area_trimmed['comboID'].str.contains(name_media)]

    abx_med_df['comboR'] = abx_med_df['comboID'].str.split(',').str[0]
    abx_med_df['comboL'] = abx_med_df['comboID'].str.split(',').str[1]

    abx_med_df['abxR'] = abx_med_df['comboR'].str.split('_').str[0]
    abx_med_df['abxL'] = abx_med_df['comboL'].str.split('_').str[0]

    abx_med_df['concR'] = abx_med_df['comboR'].str.split('_').str[-1]
    abx_med_df['concL'] = abx_med_df['comboL'].str.split('_').str[-1]

    abx_med_df.loc[(abx_med_df.abxR==name_media),'concR'] = '0'
    abx_med_df.loc[(abx_med_df.abxL==name_media),'concL'] = '0'
    abxs_list = np.concatenate([abx_med_df['abxR'].unique(),abx_med_df['abxL'].unique()])
    remove_idx = np.argwhere((abxs_list == name_media) | (abxs_list==name_bugs))
    abxs = np.delete(abxs_list, remove_idx)
    abxs = sorted(abxs)
    print(abxs,len(abxs))
    pp = PdfPages(out_path+'prelim_MIC_curves_norm_1Xbug_t'+str(t)+'norm2.pdf')
    for i in abxs:
        abx_drops_r = abx_med_df[(abx_med_df.comboR.str.contains(i)) & (abx_med_df.comboL.str.contains(name_media))]
        abx_drops_l = abx_med_df[(abx_med_df.comboR.str.contains(name_media)) & (abx_med_df.comboL.str.contains(i))]
        if abx_drops_r.shape[0]>0:
            abx_drops = abx_drops_r
            abx_drops.sort_values(by='concR',inplace=True)
            med = abx_drops.groupby('concR').median()
            plt.figure()
            sns.scatterplot(data=abx_drops, x="concR", y='t'+str(t)+'_normbugs', hue="concR",alpha=0.3)
            plt.errorbar(med.index,med['t'+str(t)+'_normbugs'],yerr=med['t'+str(t)+'_normbugs_SE'],capsize=3)
            plt.title(i)
            plt.ylim([0,2.5])
            plt.legend(bbox_to_anchor=(1.2,1))
            plt.savefig(pp, format='pdf',bbox_inches='tight',dpi=300)
        elif abx_drops_l.shape[0]>0:
            abx_drops = abx_drops_l
            abx_drops.sort_values(by='concL',inplace=True)
            med = abx_drops.groupby('concL').median()
            plt.figure()
            sns.scatterplot(data=abx_drops, x="concL", y='t'+str(t)+'_normbugs', hue="concL",alpha=0.3)
            plt.errorbar(med.index,med['t'+str(t)+'_normbugs'],yerr=med['t'+str(t)+'_normbugs_SE'],capsize=3)
            plt.title(i)
            plt.ylim([0,2.5])
            plt.legend(bbox_to_anchor=(1.2,1))
            plt.savefig(pp, format='pdf',bbox_inches='tight',dpi=300)
    pp.close()

# def plot_MIC(d_area_trimmed,out_path,t,name_media,name_bugs):
    # # abx_med_df = d_area_trimmed[d_area_trimmed['comboID'].str.contains(name_media)]
    # abx_med_df = d_area_trimmed[(d_area_trimmed['comboID'].str.contains(name_media)) & ~(d_area_trimmed['comboID'].str.contains(name_media+','+name_media))
    #                             & ~(d_area_trimmed['comboID'].str.contains(name_bugs))]
    # abx_med_df['new_comboID'] = abx_med_df['comboID'].replace({','+name_media:'',name_media+',':''},regex=True)
    # abx_med_df['abx'] = abx_med_df['new_comboID'].str.split('_').str[0]
    # abx_med_df['conc'] = abx_med_df['new_comboID'].apply(lambda x: int(x[-1]))
    # abxs_list = abx_med_df['abx'].unique()
    # remove_idx = np.argwhere((abxs_list == name_media) | (abxs_list==name_bugs))
    # abxs = np.delete(abxs_list, remove_idx)
    # # print(abxs)
    # abxs = sorted(abxs)
    #
    # pp = PdfPages(out_path+'prelim_MIC_curves_norm_1Xbug_t'+str(t)+'norm2.pdf')
    # for i in abxs:
    #     abx_drops_r = abx_med_df[(abx_med_df.Label_right.str.contains(i)) & (abx_med_df.Label_left.str.contains(name_media))]
    #     abx_drops_l = abx_med_df[(abx_med_df.Label_right.str.contains(name_media)) & (abx_med_df.Label_left.str.contains(i))]
    #     if abx_drops_r.shape[0]>0:
    #         abx_drops = abx_drops_r
    #         # abx_drops['conc'] = abx_drops.Label_right.apply(lambda x: int(x[-1]))
    #         med = abx_drops.groupby('conc').median()
    #         plt.figure()
    #         sns.scatterplot(data=abx_drops, x="conc", y='t'+str(t)+'_normbugs', hue="conc",alpha=0.3)
    #         plt.errorbar(med.index,med['t'+str(t)+'_normbugs'],yerr=med['t'+str(t)+'_normbugs_SE'],capsize=3)
    #         plt.title(i)
    #         plt.ylim([0,2.5])
    #         plt.legend(bbox_to_anchor=(1.2,1))
    #         plt.savefig(pp, format='pdf',bbox_inches='tight',dpi=300)
    #     elif abx_drops_l.shape[0]>0:
    #         abx_drops = abx_drops_l
    #         # abx_drops['conc'] = abx_drops.Label_left.apply(lambda x: int(x[-1]))
    #         med = abx_drops.groupby('conc').median()
    #         plt.figure()
    #         sns.scatterplot(data=abx_drops, x="conc", y='t'+str(t)+'_normbugs', hue="conc",alpha=0.3)
    #         plt.errorbar(med.index,med['t'+str(t)+'_normbugs'],yerr=med['t'+str(t)+'_normbugs_SE'],capsize=3)
    #         plt.title(i)
    #         plt.ylim([0,2.5])
    #         plt.legend(bbox_to_anchor=(1.2,1))
    #         plt.savefig(pp, format='pdf',bbox_inches='tight',dpi=300)
    # pp.close()
