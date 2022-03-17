import pandas as pd
import networkx as nx
from scipy import stats

##global_efficiency(G) is MODIFIED CODE FROM NETWORKX SOURCE CODE
#from networkx, I modified for weighted shortest paths -> weighted global efficiency
def global_efficiency(G):
    """Returns the average global efficiency of the graph.

    The *efficiency* of a pair of nodes in a graph is the multiplicative
    inverse of the shortest path distance between the nodes. The *average
    global efficiency* of a graph is the average efficiency of all pairs of
    nodes [1]_.

    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G, weight = 'weight')
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        g_eff = 0
    # TODO This can be made more efficient by computing all pairs shortest
    # path lengths in parallel.
    return g_eff

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
    cov = t1_covariates.loc[subject_number,:]

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