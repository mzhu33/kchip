from pims import ND2_Reader
import skimage.io as io
import os
import pandas as pd
import math as mt
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Exports nd2 file into individual tif filters
# either by enumerated each image as #0 #1 etc.
# or by rectangle enumeration (kchip) stlye _[row]_[col]
# python export_nd2.py --nd2file ../data/20211229_ab1_cpp1/20211229_110730_852/ChannelA647_7p_1X,A594_0.1p_1X,A555_1p_1X,GFP_Seq0000.nd2 --savepath ../output/nd2_export1/ --root-name nonmulti --rectangle 1 --rows 5 --cols 8

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nd2file",
                        help="path to nd2 file",
                        type=str)
    parser.add_argument("--savepath",
                        help="save path for exported tif files",
                        type=str)
    parser.add_argument("--root-name",
                        help="root name for exported tif files",
                        type=str)
    parser.add_argument("--rectangle",
                        help="0 for saving enumerated file \
                        names or 1 for rectangle styled kchip names",
                        type=int)
    parser.add_argument("--rows",
                        default=1,
                        help="# of rows of image tiling",
                        type=int)
    parser.add_argument("--cols",
                        default=1,
                        help="# of cols of image tiling",
                        type=int)
    args = parser.parse_args()
    return args

def main(nd2file,savepath,root_name,rectangle,rows,cols):
    with ND2_Reader(nd2file) as nd2:
        nd2.iter_axes = 'm'
        nd2.bundle_axes = ['c','y','x']

        if rectangle == 1:
            print('rectangle style saving now')
            df = pd.DataFrame(index=np.arange(1,len(nd2)+1,1))
            df['root_name'] = root_name+'_'
            df['tile_id'] = df.index
            df['tile_row'] = df.tile_id.apply(lambda x: int(mt.ceil(float(x)/cols)))
            df['tile_col'] = df.tile_id.apply(lambda x: int(mt.ceil(float(x)-((mt.ceil(float(x) / cols)-1)*cols))))
            df['tile_row_odd'] = df['tile_row'] % 2
            df['tile_col_new'] = np.zeros
            df['tile_col_new'][df['tile_row_odd']==1] = df['tile_col']
            df['tile_col_new'][df['tile_row_odd']==0] = cols+1-df['tile_col']
            df['new_filename'] = df['root_name']+df['tile_row'].astype(str)+'_'+df['tile_col_new'].astype(str)+'.tif'
            num=1
            for fov in nd2[:]:
                filename = df.loc[num,'new_filename']
                io.imsave(os.path.join(savepath,filename),fov)
                num+=1
        elif rectangle == 0:
            print('enumerated style saving now')
            num=1
            for fov in nd2[:]:
                filename = root_name+'_#'+str(num)+'.tif'
                io.imsave(os.path.join(savepath,filename),fov)
                num+=1

if __name__=="__main__":

    start_time = time.time()
    args = get_args().__dict__

    main(**args)
    print("--- %s seconds ---" % (time.time() - start_time))
