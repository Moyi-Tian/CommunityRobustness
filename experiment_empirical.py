import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.metrics.cluster import normalized_mutual_info_score
from clusim.clustering import Clustering
import clusim.sim as sim

import time
from multiprocessing import Pool, Lock
from multiprocessing.pool import ThreadPool

from clustering import clustering, consensus_alg
from miscellaneous_funcs import create_folder_if_needed


class exp_empirical():
	def __init__(self, network_name, network_file_path, out_directory_path, method):
		"""
        @param: network_name (str). Name of the network.
        @param: network_file_path (str). Path to the network data.
        @param: out_directory_path (str). The path where the output to be saved to.
		@param: method (str). Community detection method.
        """
		self.network_name = network_name
		self.network_file_path = network_file_path
		self.out_directory_path = out_directory_path
		self.method = method


	def __call__(self):
		
		# import network data file
		# the provided dataset should have 3 columns: 'From', 'To', 'Time'
		# the nodes ids are consecutive integers from 0; there are no repeated edges;
		# rows are sorted by timestamp
		df = pd.read_csv(self.network_file_path)
		num_node = max(max(df['From']), max(df['To'])) + 1

		# list of edge tuples
		edge_list = list(zip(df["From"],df["To"]))

		# find when every node in the network got at least one incident edge
		check_set = set()
		start_ind = 0

		for index, pair in enumerate(edge_list):
			if pair[0] not in check_set:
				check_set.add(pair[0])
			if pair[1] not in check_set:
				check_set.add(pair[1])

			start_ind = index
			if len(check_set) == num_node:
				break

		# Print network information
		print('Total number of nodes: ', num_node)
		print('Total number of edges:', len(edge_list))
		print('Initial network has ', start_ind+1, 'edges')


		# Request user inputs
		max_num_steps = len(edge_list) - (start_ind+1)
		print("max number of steps = ", max_num_steps)


		# number of steps
		num_steps = int(input("How many time steps to slice? (default max/10): ") or f"{max_num_steps // 10}")
		if num_steps > max_num_steps:
			raise ValueError('Invalid Input: Steps exceeds maximal possible value.')
		# number of partitions for consensus algorithm
		np_init = input("Number of partitions for initial network? (default value 20): ") or "20"
		# parameter to be used for clustering at time steps
		realizations = int(input("Number of realizations per time step? (default value 10): ") or "10")


		tic1 = time.time()


		# find out initial partitions
		save_path = self.out_directory_path + self.network_name + "_"
		save_path_initial_par = save_path + "initial_partitions_np_" + np_init + "_" + self.method + "/"

		tic2 = time.time()
		print('Running initial consensus...')
		consensus_init = consensus_alg(edge_list[:start_ind+1], self.method, np_init, save_path_initial_par)
		true_labels_list = consensus_init() # list of numpy arrays
		toc2 = time.time()


		# Store partition and network information for the network at initial time
		# initial number of communities mean and std
		num_comm_init = []
		for labels in true_labels_list:
			num_comm_init.append(max(labels))
		num_comm_init_mean = np.mean(num_comm_init)
		num_comm_init_std = np.std(num_comm_init)

		# distribution of community sizes
		com_size_init = [] # to create a list of lists with entries corresponding to the number of members in the community label index+1 of inner list
		for i, lis in enumerate(true_labels_list):
			com_size_init.append(np.histogram(lis, bins = num_comm_init[i])[0].tolist()) 
		# number of communities may not be the same, make them the same length
		max_num_comm_init = max(num_comm_init)
		# Then append 0s
		for com_d in com_size_init:
			for _ in range(max_num_comm_init - len(com_d)):
				com_d.append(0)
		comm_size_d_init_mean = np.mean(com_size_init, axis = 0).tolist()
		comm_size_d_init_std = np.std(com_size_init, axis = 0).tolist()

		# create networkx graph
		G_init = nx.Graph()
		G_init.add_edges_from(edge_list[:start_ind+1])

		# modularity mean and std
		modu_init = []
		for ground_label in true_labels_list:
			dic = {}
			part = []
			for id, membership in enumerate(ground_label):
				dic.setdefault(membership,[]).append(id)
			for key in dic:
				part.append(set(dic.get(key)))
			modu_init.append(nx_comm.modularity(G_init, part))
		modu_init_mean = np.mean(modu_init)
		modu_init_std = np.std(modu_init)


		# degree histogram of the inital network
		degree_d_init = nx.degree_histogram(G_init)


		# Save dataframe for initial network's information as json to folder 
		save_dic_init = {"modularity_mean": modu_init_mean, "modularity_std": modu_init_std, "number_communities_mean": num_comm_init_mean, "number_communities_std": num_comm_init_std, "community_size_mean": comm_size_d_init_mean, "community_size_std": comm_size_d_init_std, "degree_distribution": degree_d_init}

		with open(save_path_initial_par + "initial_network+partition_info.json", 'w', encoding='utf-8') as fl:
			json.dump(save_dic_init, fl, ensure_ascii=False, indent=4)


		# folder path to store clustering results for time steps
		save_path_timesteps = save_path + "s_" + f"{ num_steps }" + "/"

		print('Now moving forward in time...')
		step_size = max_num_steps//num_steps
		print("number of steps = ", num_steps)
		print("step size = ", step_size)

		# if network files for this num_steps not existed, create and save edgelist txt files to be used
		if not os.path.exists(save_path_timesteps):
			create_folder_if_needed(save_path_timesteps[:-1])


		argument_list = []
		for trial in range(num_steps):
			argument_list.append([save_path_timesteps, edge_list, num_steps, step_size, start_ind, trial, true_labels_list, realizations])
		
		def init(lock):
			global starting
			starting = lock    


		pool = ThreadPool(processes=min(num_steps, 5), initializer=init, initargs=[Lock()])
		time_step, percent_edges_added, nmi_means, nmi_stds, ele_sim_means, ele_sim_stds, modularity_means, modularity_stds, num_community_means, num_community_stds, community_size_distribution_means, community_size_distribution_stds, deg_distributions = zip(*tqdm(pool.imap(self.run_timestep, argument_list)))
		pool.close()
		pool.join()


		# save to json
		save_dic = {"time_step": time_step, "percent_edges_added": percent_edges_added, "nmi_means": nmi_means, "nmi_stds": nmi_stds, "element_sim_means": ele_sim_means, "element_sim_stds": ele_sim_stds, "modularity_mean": modularity_means, "modularity_std": modularity_stds, "number_communities_mean": num_community_means, "number_communities_std": num_community_stds, "community_size_mean": community_size_distribution_means, "community_size_std": community_size_distribution_stds, "degree_distribution": deg_distributions}

		with open(save_path_timesteps + "results_" + self.network_name + "_np_" + f"{ np_init }" + "_s_" + f"{ num_steps }" + "_r_" + f"{ realizations }" + "_" + self.method + ".json", 'w', encoding='utf-8') as f:
			json.dump(save_dic, f, ensure_ascii=False, indent=4)


		toc1 = time.time()
		print("Initial consensus done in {:.4f} seconds".format(toc2-tic2))
		print("Total time: {:.4f} seconds".format(toc1-tic1))




	def run_timestep(self, arguments):
		'''
		Run empirical experiment at time step and return lists of results
        :param arguments: a list of parameters
            arguments[0]: save_path_timesteps - File path to save network files and results
            arguments[1]: edge_list - Edge list of the whole network
            arguments[2]: num_steps - Total number of steps
            arguments[3]: step_size - Step size, i.e., number of edges added at each time step
			arguments[4]: start_ind - The index up to which the network initially becomes connected without isolated nodes
			arguments[5]: trial - The index of this trial
			arguments[6]: true_labels_list - List of community partitions from the initial consensus
			arguments[7]: realizations - Number of independent realizations at each time step
        :return: trail index, percent of edges added, NMI mean, NMI std, modularity mean, modularity std, mean of number of communities, std of number of communities, mean of list of # of members in each community, std of list of # of members in each community, degree histagram
		'''

		save_path_timesteps = arguments[0]
		edge_list = arguments[1]
		num_steps = arguments[2]
		step_size = arguments[3]
		start_ind = arguments[4]
		trial = arguments[5]
		true_labels_list = arguments[6]
		realizations = arguments[7]

		# save df up to time step as file, then define network_file_path to pass on
		save_path_timesteps_method = save_path_timesteps + self.network_name + "_s_" + f"{ num_steps }" + "_t_" + f"{ trial+1 }" + "_" + self.method + "/"
		network_file_path = save_path_timesteps + "network_" + self.network_name + "_s_" + f"{ num_steps }" + "_t_" + f"{ trial+1 }" + ".dat"

		if not os.path.exists(save_path_timesteps_method):
			create_folder_if_needed(save_path_timesteps_method[:-1])
		
		if not os.path.exists(network_file_path):
			df = pd.DataFrame(edge_list[:start_ind+(trial+1)*step_size+1])
			df.columns = ["id_left", "id_right"]
			df.sort_values(by=["id_left", "id_right"], inplace=True)
			df.to_csv(network_file_path, index=False, header=False, sep="\t")

		# run each independent realization through multiprocessing
		args = []
		for r in range(realizations):
			args.append([true_labels_list, network_file_path, num_steps, trial, r, save_path_timesteps_method])

		pool_sub = Pool(processes = min(realizations,10))
		nmi_lists, element_sim_lists, modularity_list, num_community_list, member_d_list = zip(*pool_sub.map(self.run_single_realization, args))
		pool_sub.close()
		pool_sub.join()

		nmi = [result for nmi_list in nmi_lists for result in nmi_list]
		element_sim = [r for element_sim_list in element_sim_lists for r in element_sim_list]

		# Number of communities in each realization may not be the same. In order to get the mean community sizes, we need to first make the length of these lists the same, specifically, append 0s to the shorter ones
		max_num = 0
		# First find the max length
		for comm in member_d_list:
			max_num = max(max_num, len(comm))
		# Then append 0s
		for com in member_d_list:
			for _ in range(max_num - len(com)):
				com.append(0)

		# create networkx graph for degree distribution
		df_eg = pd.read_csv(network_file_path, names = ['From', 'To'], sep="\t")
		eg_list = list(zip(df_eg['From'], df_eg['To']))
		G_curr = nx.Graph()
		G_curr.add_edges_from(eg_list)
		degree_d = nx.degree_histogram(G_curr) # degree distribution of current graph; a list where degree is the index

		return trial+1, 100*(trial+1)*step_size/(start_ind + 1), np.mean(nmi), np.std(nmi), np.mean(element_sim), np.std(element_sim), np.mean(modularity_list), np.std(modularity_list), np.mean(num_community_list), np.std(num_community_list), np.mean(member_d_list, axis = 0).tolist(), np.std(member_d_list, axis = 0).tolist(), degree_d
	



	def run_single_realization(self, arguments):
		"""
		Run a single realization at a time step

		:param arguments: a list of parameters [labels_true_list, network_file_path, trial, save_path_timesteps_method]
			arguments[0]: labels_true - The ground truth community assignment of the original network
			arguments[1]: network_file_path - path of the network at current time step
			arguments[2]: num_steps - total number of steps
			arguments[3]: trial - which step it is at out of total number of steps
			arguments[4]: r - realization index for experiment at current step
			arguments[5]: save_path_timesteps_method - path to be used to determine output path for clustering results

		:return: list of NMIs & Element Centric Similarity of the trial with respect to time 0 community assignment, modularity, number of communities, list of # of members in each community
		"""
		true_labels_list = arguments[0]
		network_file_path = arguments[1]
		num_steps = arguments[2]
		trial = arguments[3]
		r = arguments[4]
		save_path_timesteps_method = arguments[5]

		
		folder_name = self.network_name  + "_s_" + f"{ num_steps }" + "_t_" + f"{ trial+1 }" + "_r_" + str(r+1) + "_" + self.method
		output_clustering_result_path = save_path_timesteps_method + folder_name

		if self.method == "lpm":
			alg = "label"
		else:
			alg = self.method

		run_clustering_alg = clustering(network_file_path, output_clustering_result_path, folder_name, alg)
		run_clustering_alg()
				
		df = pd.read_csv(output_clustering_result_path + "/" + folder_name + ".dat", sep="\t", header=None)
		df.columns = ["id", "label"]
		df.sort_values(by=["id"], inplace=True)
		labels_curr = df["label"].to_numpy()

		nmi = []
		element_sim = []
		for true_labels in true_labels_list:
			# calculate nmi and clusim
			c1 = Clustering()
			c1.from_membership_list(true_labels)
			c2 = Clustering()
			c2.from_membership_list(labels_curr)
			element_sim.append(sim.element_sim(c1, c2, alpha = 0.9))
			nmi.append(sim.nmi(c1, c2, norm_type='sum'))
			# nmi.append(normalized_mutual_info_score(true_labels, labels_curr))
		

		# get number of communities and distribution of community sizes
        # Note: membership ids are from 1; node ids are from 0
		num_community = max(df["label"])
		member_d = df.groupby(by = "label").count()["id"].tolist() # list of number of members in each community in order from community label 1 to num_community

        # create networkx graph; compute modularity; get degree histagram
		df_edge = pd.read_csv(network_file_path, names = ['From', 'To'], sep="\t")
		edge_list = list(zip(df_edge['From'], df_edge['To']))
		G = nx.Graph()
		G.add_edges_from(edge_list)

		df_member = df.groupby(by = 'label')['id'].apply(set).reset_index(name='member_ids') # dataframe with columns 'label' and 'member_ids'. 'member_ids' is a set of node ids corresponding to the community label
		part = df_member['member_ids'].tolist() # list of sets where each set gives the members' ids
		modularity = nx_comm.modularity(G, part)

		return nmi, element_sim, modularity, num_community, member_d
