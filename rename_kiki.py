import os
import pandas as pd
import numpy as np
import math as mt

def find_images(path_to_dir, suffix=".tif" ):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) ]

def rename_kiki(path_to_dir,split_str,rows,columns):
    '''
    split_str: prefix to the image number e.g. "#" or "XY" 
    '''
    files = find_images(path_to_dir)
    df = pd.DataFrame(data=files,columns=['file_name'])

    df['root_name'] = [i.split(split_str)[0] for i in files]
    df['tile_id'] = [i.split(split_str)[1].split('.')[0] for i in files]
    df = df.sort_values(by='tile_id')
    df['tile_row'] = df.tile_id.apply(lambda x: int(mt.ceil(float(x)/columns)))
    df['tile_col'] = df.tile_id.apply(lambda x: int(mt.ceil(float(x)-((mt.ceil(float(x) / columns)-1)*columns))))
    df['tile_row_odd'] = df['tile_row'] % 2
    df['tile_col_new'] = np.zeros
    df['tile_col_new'][df['tile_row_odd']==1] = df['tile_col']
    df['tile_col_new'][df['tile_row_odd']==0] = columns+1-df['tile_col']
    df['new_filename'] = df['root_name']+df['tile_row'].astype(str)+'_'+df['tile_col_new'].astype(str)+'.tif'
    for j in enumerate(df.file_name):
        src = os.path.join(path_to_dir,j[1])
        dst = os.path.join(path_to_dir,df.iloc[j[0],-1])
        os.rename(src,dst)
    return df
