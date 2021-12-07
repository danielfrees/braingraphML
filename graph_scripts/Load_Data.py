import numpy as np
import pandas as pd

def load_mats(data, ROIS):
    print(f"Shape of passed ROI-ROI matrix data is: {np.shape(data['corrs'])}")
    print(f"Number of ROI indices passed is: {np.shape(ROIS['ROIS'])}")
     
    ROI_labels = []
    for i in range(np.shape((ROIS['ROIS']))[1]):
         ROI_labels.append(ROIS['ROIS'][:,i][0][0])
            
    mats = []
    #iterate over 3rd dimension(each participant, and grab their corr matrix)
    for i in range(data['corrs'].shape[-1]):
        #load correlations for given participant and label according to ROIs
        patient_conn = pd.DataFrame((data['corrs'][:,:,i]), columns = ROI_labels, index = ROI_labels)
        
        #mark all correlations of ROI to self (originally NaN) as weight 0.0 (standard in the literature)
        patient_conn.fillna(0, inplace = True)
        
        #future: change to different correlation statistic? Filter out super low correlations? etc. 
        
        #filter for 264 atlas ROIs
        patient_conn = patient_conn.filter(regex='264', axis = 1)
        patient_conn = patient_conn.filter(regex='264', axis = 0)
        
        mats.append(patient_conn)
        
    print(f"All matrices filtered for '264' labeled atlas ROIs only. Shape of each participant matrix is {np.shape(mats[0])}.")
    
    return mats