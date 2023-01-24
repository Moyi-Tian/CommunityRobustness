import os
import csv
import json
import numpy as np

from tqdm import tqdm
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from clusim.clustering import Clustering
import clusim.sim as sim

from time import sleep
from multiprocessing import Pool

from clustering import clustering
from miscellaneous_funcs import create_folder_if_needed, add_random_edges


class exp_synthetic():
    def __init__(self, N, mu, k, maxk, minc, maxc, method, ground_truth_network_file, ground_truth_community_file, out_directory_path, edge_append_flag, add_up_to):
        """
        @param: N (int). Number of nodes.
        @param: mu (float). Mixing parameter.
        @param: k (int). Average degree.
        @param: maxk (int). Maximum degree .
        @param: minc (int). Minimun community size.
        @param: maxc (int). Maximum community size.
        @param: method (str). Community detection method.
        @param: ground_truth_network_file (str). Path to the ground truth network file.
        @param: ground_truth_community_file (str). Path to the ground truth community file.
        @param: out_directory_path (str). The path where the output to be saved to.
        @param: edge_append_flag(int). The way selecting appended edges at random.
        @param: add_up_to(float). Times of original number of edges the algorithm adds up to by the end.
        """
        self.N = N
        self.mu = mu
        self.k = k
        self.maxk = maxk
        self.minc = minc
        self.maxc = maxc
        self.method = method
        self.ground_truth_network_file = ground_truth_network_file
        self.ground_truth_community_file = ground_truth_community_file
        self.out_directory_path = out_directory_path
        self.edge_append_flag = edge_append_flag
        self.add_up_to = add_up_to


    def __call__(self):
        
        # parameter lists to be saved
        means_nmi = []
        stds_nmi = []
        means_element_sim = []
        stds_element_sim = []
        perc_added_edges = []

        # Get ground truth to a numpy array
        df = pd.read_csv(self.ground_truth_community_file, sep="\t", header=None)
        df.columns = ["id", "label"]
        df.sort_values(by=["id"], inplace=True)
        labels_true = df["label"].to_numpy()

        # Load the seed network
        with open(self.ground_truth_network_file) as f:
            reader = csv.reader(f, delimiter="\t")
            edges = set([tuple(map(int, rec)) for rec in reader])

        if self.N > 50 and len(edges) > 100:
            max_number_edges = int(round(self.add_up_to*len(edges), -3)) # Nearest maximum number of edges to be added close to add_up_to*100%
            step = int(max_number_edges/50)
        else:
            max_number_edges = len(edges) # if network is too small, add up to double of current number of edges
            step = int(max_number_edges/2)

        realizations = 50 # Number of repetitions to add the same number of edges

        # Add random edges and compute statistics
        for number_edges in tqdm(range(0, max_number_edges+step, step)):

            argument_list = []

            for tr in range(0, realizations):
                argument_list.append([labels_true, edges, number_edges, tr])

            pool = Pool(processes = min(30, realizations))
            nmi, element_sim = zip(*pool.map(self.trial, argument_list))
            pool.close()
            pool.join()

            # Aggregate statistics
            means_nmi.append(np.mean(nmi))
            stds_nmi.append(np.std(nmi))
            means_element_sim.append(np.mean(element_sim))
            stds_element_sim.append(np.std(element_sim))
            perc_added_edges.append((100*number_edges)/len(edges))

        save_dic = {"percent_edges_added": perc_added_edges, "nmi_means": means_nmi, "nmi_stds": stds_nmi, "element_sim_means": means_element_sim, "element_sim_stds": stds_element_sim}
        
        # save json data
        with open(self.out_directory_path + "results_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + "_maxk_" + str(self.maxk) + "_minc_" + str(self.minc) + "_maxc_" + str(self.maxc) + "_f_" + str(self.edge_append_flag) + "_add_" + str(self.add_up_to) + "_" + self.method + ".json", 'w', encoding='utf-8') as f:
            json.dump(save_dic, f, ensure_ascii=False, indent=4)

    

    def trial(self, arguments):
        """
        Run a single realization for a specified number of edges added to the network

        :param arguments: a list of parameters [labels_true, edges, number_edges, trial]
            arguments[0]: labels_true - The ground truth community assignment of the original network
            arguments[1]: edges - Edge list of the original network
            arguments[2]: number_edges - Number of edges to be added in this single realization
            arguments[3]: trial - The index of this trial

        :return: NMI & element-centric similarity of the trial with respect to the original community assignment, modularity, number of communities, list of # of members in each community, degree histagram
        """
        labels_true = arguments[0]
        edges = arguments[1]
        number_edges = arguments[2]
        trial = arguments[3]

        # Check if perturbed_network exists: if not, create new network and save
        network_file_path = self.out_directory_path + "perturbed_data/" + "network_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + "_e_" + str(number_edges) + "_t_" + str(trial+1) + "_f_" + str(self.edge_append_flag) + "_add_" + str(self.add_up_to) + ".dat"
        if not os.path.exists(network_file_path):
            # Add edges
            new_network = add_random_edges(edges, number_edges, labels_true, self.edge_append_flag)
            df = pd.DataFrame(new_network)
            df.columns = ["id_left", "id_right"]
            df.sort_values(by=["id_left", "id_right"], inplace=True)
            df.to_csv(network_file_path, index=False, header=False, sep="\t")

        # Compute clustering
        # Check if the clustering result folder exists: if not, create such folder; then run the clustering algorithm
        output_clustering_path_name = "clustering_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + "_f_" + str(self.edge_append_flag) + "_add_" + str(self.add_up_to) + "_" + self.method
        
        if not os.path.exists(self.out_directory_path + "perturbed_data/" + output_clustering_path_name):
            create_folder_if_needed(self.out_directory_path + "perturbed_data/" + output_clustering_path_name)
        
        folder_name = "clustering_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + "_e_" + str(number_edges) + "_t_" + str(trial+1) + "_f_" + str(self.edge_append_flag) + "_add_" + str(self.add_up_to) + "_" + self.method
        output_clustering_path = self.out_directory_path + "perturbed_data/" + output_clustering_path_name + "/" + folder_name

        run_clustering_alg = clustering(network_file_path, output_clustering_path, folder_name, self.method)
        run_clustering_alg()

        # sleep if clustering result is not ready
        while not os.path.exists(output_clustering_path + "/" + folder_name + ".dat"):
            sleep(1)
        
        # load membership results
        df = pd.read_csv(output_clustering_path + "/" + folder_name + ".dat", sep="\t", header=None)
        df.columns = ["id", "label"]
        df.sort_values(by=["id"], inplace=True)
        labels_pred = df["label"].to_numpy()

        # calculate clusim
        c1 = Clustering()
        c1.from_membership_list(labels_true)

        c2 = Clustering()
        c2.from_membership_list(labels_pred)

        element_sim = sim.element_sim(c1, c2, alpha = 0.9)

        # return NMI, element-centric similarity
        return normalized_mutual_info_score(labels_true, labels_pred), element_sim
