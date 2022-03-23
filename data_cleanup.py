import pandas as pd
import numpy as np

def derep(droplets,condensed):
    # merge your droplets and condensed df's and it going to dereplicate "left" and "right"
    merged = condensed[condensed.Total==2].merge(droplets[['Hash','Label']],on=['Hash'],how='outer')
    merged_sort = merged.groupby('Hash', as_index=False).agg({'Label':lambda x: ','.join(x)})
    merged_sort['Label'] = merged_sort['Label'].apply(lambda x: sorted(x.split(',')))
    merged_sort['Label_left'] = merged_sort['Label'].apply(lambda x: x[0])
    merged_sort['Label_right'] = merged_sort['Label'].apply(lambda x: x[1])
    left = pd.merge(merged_sort[['Hash','Label_left']],droplets,how='left', left_on=['Hash','Label_left'],right_on=['Hash','Label'])
    right = pd.merge(merged_sort[['Hash','Label_right']],droplets,how='left', left_on=['Hash','Label_right'],right_on=['Hash','Label'])
    merge_derep = left.merge(right,on=['Hash'],how='outer',suffixes=('_left','_right'))
    merge_derep = merge_derep.loc[:,~merge_derep.columns.duplicated()]
    trimmed = merge_derep.merge(condensed,on=['Hash'],how='outer')
    return trimmed

def calc_QC(trimmed,tp):
    # Calculates centroids and barcode distances
    trimmed['CentroidX_left'] = trimmed.groupby('Label_left')['PlaneX_left'].transform('median')
    trimmed['CentroidY_left'] = trimmed.groupby('Label_left')['PlaneY_left'].transform('median')
    trimmed['distance_left'] = np.sqrt((trimmed.PlaneX_left-trimmed.CentroidX_left)**2\
                                       +(trimmed.PlaneY_left-trimmed.CentroidY_left)**2)

    trimmed['CentroidX_right'] = trimmed.groupby('Label_right')['PlaneX_right'].transform('median')
    trimmed['CentroidY_right'] = trimmed.groupby('Label_right')['PlaneY_right'].transform('median')
    trimmed['distance_right'] = np.sqrt((trimmed.PlaneX_right-trimmed.CentroidX_right)**2\
                                        +(trimmed.PlaneY_right-trimmed.CentroidY_right)**2)

    # renames columns
    trimmed = trimmed.rename(columns = {'Dye 0_left':'Dye0_left', \
                                        'Dye 1_left':'Dye1_left', \
                                        'Dye 2_left':'Dye2_left', \
                                        'Dye 0_right':'Dye0_right',\
                                        'Dye 1_right':'Dye1_right',\
                                        'Dye 2_right':'Dye2_right'})
    # normalized reporter signal
    trimmed['summed']=(trimmed.Dye0_left+trimmed.Dye1_left+trimmed.Dye2_left+\
                       trimmed.Dye0_right+trimmed.Dye1_right+trimmed.Dye2_right)
    med = trimmed.summed.median()
    trimmed['ratio']=trimmed.summed/med

    for i in range(tp):
        trimmed['t'+str(i)+'_norm2']=trimmed['t'+str(i)]/trimmed.ratio

    return trimmed
