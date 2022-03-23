# From bchip libraries
from scipy.spatial.distance import pdist, squareform
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def droplets_in_same_well(positions,config):
    '''Identify droplets that are in the same well based on distance threshold
    Inputs:
    - Positions, (k x 2 ndarray), [x, y] positions of droplet centers
    - close_threshold, max distance between droplet centers to be considred in same well (default = 25)
    Outputs:
    - list of tuples (index 1, index 2) of indexes in positions of droplets in the same well
    '''

    pair_distances = squareform(pdist(positions))
    pair_distances[pair_distances==0]=float('inf')
    ## OLD METHOD
    # close_threshold = int(160*config['image']['objective']/config['image']['bins']/config['image']['pixel_size']+10)
    # droplet1,droplet2 = np.where(pair_distances<close_threshold)
    # zip_list = zip(droplet1,droplet2)

    droplet1 = np.asarray(range(pair_distances.shape[0]))
    droplet2 = np.argmin(pair_distances,axis=0)
    zip_list = zip(droplet1,droplet2)

#     Sort list so small index is always first in tuple, and remove duplicates
    return list(set([tuple(sorted(tup_)) for tup_ in zip_list]))

def remove_overlap(config,df,show=0):
    ''' Remove droplets found in regions of image that overlap.
    Inputs:
        - config, config dictionary
        - df, pandas dataFrame
    Returns:
        - copy of dataframe, with dropped rows
    '''
    df = df.copy()
    maxX = df['IndexX'].max()
    maxY = df['IndexY'].max()

    overlap = config['image']['size']*(1-config['image']['overlap'])

    rmv_index = ((df['ImageX'] > overlap) & ~(df['IndexX']==maxX)) | ((df['ImageY'] > overlap) & ~(df['IndexY']==maxY))

    if show:
        print ('Removed: ' + str(np.sum(rmv_index)) + ' wells from dataFrame due to overlap in images.')

    return df.drop(df.index[rmv_index]).reset_index(drop=True)

def hashdrops(droplets,config):
    unique_images = {tuple(row) for row in droplets[['IndexX','IndexY']].values}

    droplets['Well_ID']=0
    for x,y in unique_images:
        d = droplets.loc[(droplets['IndexX']==x) & (droplets['IndexY']==y)]
        positions = d[['ImageX','ImageY']].values
        same_well = droplets_in_same_well(positions,config)

        well_id = np.zeros((d.shape[0],1))

        for i,dab in enumerate(same_well):
            d_a, d_b = dab
            well_id[d_a]=i
            well_id[d_b]=i

        droplets.loc[(droplets['IndexX']==x) & (droplets['IndexY']==y),'Well_ID']=well_id

    # Remove well_id = 0
    droplets = droplets[droplets['Well_ID']!=0]

    # apply hash
    droplets['Hash'] = droplets.apply(lambda row: hash((row['IndexX'],row['IndexY'],row['Well_ID'])),axis=1)
    return droplets
