# Common functions

import os
import errno
import shutil
import random
import numpy as np


def create_folder_if_needed(out_dir_path):
    """
    Creates a non existing folder
    @param: out_dir_path (str). The path where the folder is created
    @return: None
    """

    try:
        os.makedirs(out_dir_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise



def delete_folder_if_needed(directory_path):
    """
	Delete a folder if exist. Clustering algorithm will not run if already computed
	@param: directory_path (str). Path of the directory to delete
	@return: None
	"""
    try:
        shutil.rmtree(directory_path,ignore_errors=True)
    except:
        pass




### Functions for perturbating edges###

def propose_edge(max_index):
    """
    Come up with a random edge to be added avoiding self-loops
    @param: max_index (int). Bigest node id in the edgelist
    @return: (left_side, right_side) (tuple). Proposed new edge
    """
    
    left_side = random.randint(0, max_index)
    right_side = random.randint(0, max_index)

    # To avoid self-loops
    while right_side == left_side:
        right_side = random.randint(0, max_index)
        
    return tuple([left_side, right_side])



def add_random_edges(edges, times, labels, flag):
    """
    Add random a certain time of random edges to an existing network
    @param: edges (set). Set of edge tuples
    @param: times (int). Number of new edges to be added
    @param: labels (numpy array). Numpy array of the membership of nodes (membership labels from 1, node ids from 0)
    @param: flag (int). Flag indicating how to select randomly appended edges (0=uniform at random; 1=across communities; 2=in same community)
    @return: new_edgelist (list). Edge list with random edges added
    """
    
    edges_list = list(edges) 
    max_index = max(max(np.array(edges_list)[:,0]), max(np.array(edges_list)[:,1]))
    count = 0 # Count the number of added edges
    
    edges_copy = edges.copy()
    
    while count < times:
        
        candidate_edge = propose_edge(max_index)
                
        # To avoid repeated edges
        if flag == 0:
            while (candidate_edge[0], candidate_edge[1]) in edges_copy or (candidate_edge[1], candidate_edge[0]) in edges_copy:
                candidate_edge = propose_edge(max_index)
        elif flag == 1:
            while (candidate_edge[0], candidate_edge[1]) in edges_copy or (candidate_edge[1], candidate_edge[0]) in edges_copy or labels[candidate_edge[0]] == labels[candidate_edge[1]]:
                candidate_edge = propose_edge(max_index)
        elif flag == 2:
            while (candidate_edge[0], candidate_edge[1]) in edges_copy or (candidate_edge[1], candidate_edge[0]) in edges_copy or labels[candidate_edge[0]] != labels[candidate_edge[1]]:
                candidate_edge = propose_edge(max_index)
    
        edges_copy.add(candidate_edge)
        edges_list.append(candidate_edge)
            
        count += 1

    new_edge_list = list(set(edges_list))
        
    return new_edge_list


