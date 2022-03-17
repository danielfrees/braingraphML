import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

def remove_atlas(str):
    str = str.replace('BN_Atlas_264_2mm_wkbrois.', '')
    return str

#Clean up ROI names
def clean_roi_names(mats):
    for i in range(len(mats)):
        mats[i] = mats[i].rename(remove_atlas, axis=1)
        mats[i] = mats[i].rename(remove_atlas, axis=0)
    return mats

def pass_through_cutoff(val, cutoff):
    if val <= cutoff:
        val = 0
    return val

def dynamic_threshold(mat, percentile = 75):
    '''Dynamically threshold Fisher's Z values so that only the specified 
    percentile and up of correlation values are retained. This gives all graphs a density of 
    1-percentile.

    Expects a matrix with negative correlations already removed. Supported by current rsfMRI GT Literature.'''
    arr = mat.values
    cutoff = np.percentile(arr, 75)
    for col in mat.columns:
        mat[col] = mat[col].apply(pass_through_cutoff, args = [cutoff])
    return mat

def threshold_all_mats(mats, percentile):
    for i in range(len(mats)):
        mats[i] = dynamic_threshold(mats[i], percentile)
    return mats

def zero_negatives(val):
    if val<=0:
        val=0
    return val

def process_negative_corrs(mats, method = 'zero'):
    if method == 'zero':
        for i in range(len(mats)):
            mats[i] = mats[i].applymap(zero_negatives)
    return mats
        



def limit_features(df, t_test, pval=0.25, verbose=False):
    features_to_keep = []

    for (key, value) in t_test.items():
        if value.pvalue < 0.25:
            if verbose:
                print(f"Notable Group Difference: {key}")
                print(f"p-val: {value}")
                print()
            features_to_keep.append(key)
    
    return df[features_to_keep]

def balance_sample(sample_x, sample_y, method = 'oversample', verbose = False):
    '''Function to balance an x and y sample.
    Oversampling is consistent across x and y, assuming indices are consistent.
    Can specify method to be 'oversample' or 'undersample' '''
    
    compare_indices = sample_x.index == sample_y.index
    
    if not compare_indices.all():
        print('ERROR! Indices must match between x and y samples for proper resampling!')
        return #immediately quit
    
    #reset indices so we don't ever write back over indices, 
    #causing unexpected behavior when trying to oversample
    sample_x.reset_index(inplace=True, drop = True)
    sample_y.reset_index(inplace=True, drop = True)
    
    counts = np.bincount(sample_y)
    if verbose:
        print(f'Counts: {counts}')
    num_tbi = counts[1]
    num_hc = counts[0]
    
    #kept this out of the loop so that samples don't become increasingly 
    #likely to be duplicated over and over
    #would be important with big data
    tbi_index = sample_y[sample_y==1].index
    hc_index = sample_y[sample_y==0].index
    
    
    #NOTE: HC_INDEX AND TBI_INDEX DO NOT UPDATE IN OVERSAMPLING, TO PREVENT
    #ALREADY OVERSAMPLED SAMPLES FROM BECOMING MORE LIKELY TO DUPLICATE
    if method == 'oversample': 
        while num_tbi < num_hc:
            #randomly select a tbi point and duplicate it in the sample
            to_add = np.random.choice(tbi_index)
            val_to_add = sample_y[to_add]
            series_to_add = pd.Series(val_to_add)
            sample_y = sample_y.append(series_to_add, ignore_index=True)

            #duplicate the corresponding piece of the features dataframe
            features_to_add = sample_x.loc[[to_add]]
            x_series = pd.DataFrame(features_to_add)
            sample_x = sample_x.append(x_series, ignore_index=True)

            #update counts of tbi and hc
            counts = np.bincount(sample_y)
            if verbose:
                print(f'Counts: {counts}')
            num_tbi = counts[1]
            num_hc = counts[0]

        #not needed for this dataset but included for symmetry
        while num_hc < num_tbi:
            #randomly select a hc point and duplicate it in the sample
            to_add = np.random.choice(hc_index)
            val_to_add = sample_y[to_add]
            series_to_add = pd.Series(val_to_add)
            sample_y = sample_y.append(series_to_add, ignore_index=True)

            #duplicate the corresponding piece of the features dataframe
            features_to_add = sample_x.loc[[to_add]]
            x_series = pd.DataFrame(features_to_add)
            sample_x = sample_x.append(x_series, ignore_index=True)

            #update counts of tbi and hc
            counts = np.bincount(sample_y)
            if verbose:
                print(f'Counts: {counts}')
            num_tbi = counts[1]
            num_hc = counts[0]
            
            
    #NOTE: HC_INDEX AND TBI_INDEX UPDATED WITH THE DROPPED VALS HERE TO AVOID 
    #ERRORS
    elif method == 'undersample':
        while num_tbi < num_hc:
            #randomly select an hc point and delete it
            to_del = np.random.choice(hc_index)
            sample_y = sample_y.drop(labels = to_del, axis = 0)

            #del from features as well
            sample_x = sample_x.drop(labels = to_del, axis=0)

            #update counts of tbi and hc
            counts = np.bincount(sample_y)
            if verbose:
                print(f'Counts: {counts}')
            num_tbi = counts[1]
            num_hc = counts[0]
            
            hc_index = hc_index.drop(to_del)

        #not needed for this dataset but included for symmetry
        while num_hc < num_tbi:
            #randomly select an tbi point and delete it
            to_del = np.random.choice(tbi_index)
            sample_y = sample_y.drop(labels = to_del, axis = 0)

            #del from features as well
            sample_x = sample_x.drop(labels = to_del, axis=0)

            #update counts of tbi and hc
            counts = np.bincount(sample_y)
            if verbose:
                print(f'Counts: {counts}')
            num_tbi = counts[1]
            num_hc = counts[0]
            
            tbi_index = tbi_index.drop(to_del)
        
        
    return (sample_x, sample_y)

def split_and_feature_select(all_data, target_name='mTBI', drop_names = ['Subject_number', 'Density'] verbose=False):
    '''NEATER PIPELINE FOR DATA BALANCING (FOR EASY RERUN)
    Expects a dataframe (all_data) which contains all features
    and the target variable. Can pass drop_names to indicate columns
    to be dropped.'''

    if verbose:
        print('All Data:')
        display(all_data.head())

    #Separate features from target and drop indicated drop columns
    features = all_data.drop(target_name, axis=1, inplace=False)
    target = all_data[target_name]
    features = features.drop(drop_names, axis=1)

    #MANUAL FEATURE SELECTION BASED ON TTEST
    t_test = {}

    for column in features.columns:
        t_test[column] = stats.ttest_ind(a=tbi_data[column], b=hc_data[column], equal_var = False, alternative = 'two-sided', random_state=42)

    features = limit_features(features, t_test, pval=0.05)
    
    if verbose:
        print('Features after t-test feature selection')
        display(features.head(5))

    #CONVERT TO NUMPY ARRAYS FOR SPLITTING AND MODELING
    features = features.values
    target = target.values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 0)

    #X_train, y_train = balance_sample(X_train, y_train, method = 'undersample')
    #X_test, y_test = balance_sample(X_test, y_test, method = 'undersample')

    #scale for KNN and other sensitive models
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    
    return X_train, X_test, y_train, y_test