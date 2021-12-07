import numpy as np
import pandas as pd
import networkx as nx

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



#--------DYNAMIC THRESHOLDING-----------------------------------------------------------------
#helper function to take values and determine if they are above a cutoff
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
    '''Threshold all adjacency matrices using dynamic_threshold function'''
    for i in range(len(mats)):
        mats[i] = dynamic_threshold(mats[i], percentile)
    return mats

#---------------------------------------------------------------------------------------------------


#------------------CLEANING/MAKING PRETTY---------------------------------------------------------------
def remove_atlas(str):
    str = str.replace('BN_Atlas_264_2mm_wkbrois.', '')
    return str

#Clean up ROI names
def clean_roi_names(mats):
    for i in range(len(mats)):
        mats[i] = mats[i].rename(remove_atlas, axis=1)
        mats[i] = mats[i].rename(remove_atlas, axis=0)
    return mats


#---------------------------------------------------------------------------------------------------



#-----------------------GRAPH THEORY FUNCTIONS ------------------------------------------------------------

#insert any graph and get node of interest which should work for all graphs (same ROIs)
def get_nodes_of_interest(graph):
    nodes_of_interest = []
    for node in graph.nodes:
        if 'Amy' in node:
            print(node)
            nodes_of_interest.append(node)
    return nodes_of_interest


def generate_data_for_subject(mats, t1_covariates, nodes_of_interest, subject_number):  
    '''Generates Graph Theory Data for a given subject.

    Expects list of weighted adjacency matrices for all subjects, 
    covariates matrix/array of same length as # of subjects, 
    nodes of interest for certain metrics, 
    subject number to calculate for.

    Outputs a dataframe with all graph theory data for this subject'''
    #INIT SUBJECT
    subject = pd.Series(dtype = 'int64')

    #GRAB COVARIATES
    cov = t1_covariates.loc[0,:]

    subject['Subject_Number'] = cov['conn.sub.num']
    subject['mTBI'] = cov['mTBI']
    subject['Male'] = cov['male']

    #subject

    #GENERATE GRAPH FOR THIS SUBJECT
    g = nx.from_pandas_adjacency(mats[subject_number])
    g.name = f'subj_{1}'

    #unhash if you want to display patient graph:
    #edges, weights = zip(*nx.get_edge_attributes(g,'weight').items())
    #pos = nx.spring_layout(p1)
    #nx.draw(p1, pos, node_color='black', edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.coolwarm)

    #CLUSTERING COEFFICIENTS
    subject['Avg_Clustering (W)'] = nx.average_clustering(g, weight = 'weight')
    subject['Avg_Clustering (UW)'] = nx.average_clustering(g)

    #INVESTIGATE CHARACTERISTIC PATH LENGTH FOR G
    UW_CPL = nx.average_shortest_path_length(g)

    subject['Characteristic_Path_Length'] = UW_CPL

    #INVESTIGATE DENSITY (Should be the same for all graphs!! ~1-percentile chosen for dynamic thresholding)
    subject['Density'] = nx.density(g)

    #INVESTIGATE TRANSITIVITY 
    subject['Transitivity'] = nx.transitivity(g)


    #NODES OF INTEREST-----------------------------------------------------------------------------------------------
    #NODAL DEGREE FOR NODES OF INTEREST
    #Investigate Amygdala Nodes Currently (Nodal Degree)
    degree_view = g.degree(nbunch = nodes_of_interest)
    nodes = (node for (node, val) in degree_view)
    for node in nodes:
        subject[f"Node_Degree--{node}"] = degree_view[node]


    #NODAL STRENGTH FOR NODES OF INTEREST
    degree_view = g.degree(nbunch = nodes_of_interest, weight = 'weight')
    nodes = (node for (node, val) in degree_view)
    for node in nodes:
        subject[f"Node_Strength--{node}"] = degree_view[node]


    #CALCULATE CLOSENESS CENTRALITIES FOR NODES OF INTEREST
    dict_ccs = {}
    for n in nodes_of_interest:
        dict_ccs[n] = nx.closeness_centrality(g, n)
        subject[f"Closeness_Centrality--{n}"] = nx.closeness_centrality(g, n)


    #INVESTIGATE SHORTEST PATHS BETWEEN THESE PAIRS OF NODES
    pair_1 = ['L_dAmy', 'L_MB']
    pair_2 = ['L_dAmy', 'R_MB']

    SP_pair1 = nx.shortest_path_length(g,pair_1[0],pair_1[1])
    SP_pair2 = nx.shortest_path_length(g,pair_2[0],pair_2[1])

    subject[f'Shortest_Path--{pair_1[0]}&{pair_1[1]}'] = SP_pair1
    subject[f'Shortest_Path--{pair_2[0]}&{pair_2[1]}'] = SP_pair2


    #-----------------------------------------------------------------------------------------------------------------

    #CONVERT TO INT DATATYPE WHERE APPROPRIATE AND MAKE INTO DF

    subject = pd.DataFrame(subject).transpose()

    int_data_dict = {'Subject_Number': 'int32', 
                     'mTBI': 'int32', 
                     'Male': 'int32', 
                     'Node_Degree--L_Amyg_mAmyg (medial amygdala)':'int32', 
                     'Node_Degree--R_Amyg_mAmyg (medial amygdala)':'int32', 
                     'Node_Degree--L_Amyg_lAmyg (lateral amygdala)':'int32', 
                     'Node_Degree--R_Amyg_lAmyg (lateral amygdala)':'int32', 
                     'Node_Degree--L_dAmy':'int32', 
                     'Node_Degree--R_dAmy':'int32', 
                     'Node_Degree--L_mAmy':'int32', 
                     'Node_Degree--R_mAmy':'int32', 
                     'Node_Degree--L_vlAmy':'int32', 
                     'Node_Degree--R_vlAmy': 'int32',
                     'Shortest_Path--L_dAmy&L_MB':'int32',           
                     'Shortest_Path--L_dAmy&R_MB': 'int32'}

    subject = subject.astype(int_data_dict)
    
    return subject

#----------------------------------------------------------------------------------------------------------------------------------