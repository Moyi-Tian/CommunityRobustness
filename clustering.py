import os
import errno
import subprocess
import pandas as pd
import random
from time import sleep

# For fast_consensus
import networkx as nx
import numpy as np
import igraph as ig
import community as cm
import multiprocessing as mp
import leidenalg


from miscellaneous_funcs import create_folder_if_needed, delete_folder_if_needed



class consensus_alg():

	def __init__(self, edge_list, method, np, out_directory):
		"""
        @param: edge_list(list). List of edge tuples.
        @param: method (str). Community detection method.
        @param: np (int). Number of partitions.
		@param: out_directory (str). Path to save results.
        """
		self.edge_list = edge_list
		self.method = method
		self.np = int(np)
		self.out_directory = out_directory

	def __call__(self):

		run_fast_consensus = fast_consensus(self.edge_list, self.np, self.method, self.out_directory)
		run_fast_consensus()

		label_list = []

		for file in range(int(self.np)):
		# make "file" in result folder into numpy label array 

			# print(self.out_directory + f"{ file+1 }" + " exists?", os.path.exists(self.out_directory + f"{ file+1 }"))
			
			df_cluster = pd.read_csv(self.out_directory + f"{ file+1 }", names = ['nodes'])

			# node id starts from 0, membership starts from 1
			df_out = pd.DataFrame(columns = ['id', 'membership'])

			for id, row in df_cluster.iterrows():
				nodes = row[0].split(' ')
				df_new = pd.DataFrame([[int(node), id + 1] for node in nodes], columns = ['id', 'membership'])
				df_out = pd.concat([df_out, df_new], names = ['id', 'membership'])

			df_out.sort_values(by = ['id'], inplace=True)
			label_list.append(df_out['membership'].to_numpy())

		return label_list





class clustering():
	def __init__(self, network_file_path, output_clustering_path, folder_name, method):
		"""
        @param: network_file_path(str). Path to network file.
        @param: output_clustering_path (str). Path to save clustering result.
        @param: folder_name (str). Folder name to be used to name output file.
		@param: method (str). Community detection method.
        """
		self.network_file_path = network_file_path
		self.output_clustering_path = output_clustering_path
		self.folder_name = folder_name
		self.method = method

	def __call__(self):
		if self.method == "leiden":
			create_folder_if_needed(self.output_clustering_path)
			cluster_alg = Leiden_clustering(self.network_file_path, self.output_clustering_path + "/" + self.folder_name + ".dat", self.method)
			cluster_alg()
		else:
			community_detection_methods = {"louvain": 4, "infomap": 2, "label": 5}
			community_detection_choice = community_detection_methods[self.method]
			Lancichinetti_alg = Lancichinetti_clustering(self.network_file_path, community_detection_choice, self.output_clustering_path + "/", self.folder_name)
			Lancichinetti_alg()



class Lancichinetti_clustering():
	"""
	Run clustering algorithm using Lancichinetti code
	@param: network_file_path (str). Path to the network file
	@param: community_detection_choice (int). id of the clustering program. The convention is: 0: oslom undirected, 1: oslom directed, 2: infomap undirected, 3: infomap
	directed, 4: louvain, 5: label propagation method, 6: hierarchical infomap undirected, 7: hierarchical infomap directed, 8: modularity optimization (simulated annealing zero    
	temperature)
	@param: output_clustering_path (str). The directory where the output is going to be placed
	@param: folder_name (int). Name of folder to save the output.
	@return: None
	"""
	
	def __init__(self, network_file_path, community_detection_choice, output_clustering_path, folder_name):
		self.network_file_path = network_file_path
		self.community_detection_choice = community_detection_choice
		self.output_clustering_path = output_clustering_path
		self.folder_name = folder_name

		# hyperparameters
		self.number_of_clusterings = 1
		self.clustering_program_path = os.path.dirname(os.getcwd()) + "/clustering_programs/Lancichinetti_clustering_programs/"


	def __call__(self):
		"""
		Run clustering algorithm using Lancichinetti code with specified community detection algorithm and store the results using the format (node_id>=0, membership>=0).
		Write clustering results in a file using the format (node_id>=0, memmbership>=1).
		""" 

		delete_folder_if_needed(self.output_clustering_path)

		subprocess.check_call(["time", "python3", "select.py", "-n", self.network_file_path, "-p", str(self.community_detection_choice), "-f", self.output_clustering_path, "-c", str(self.number_of_clusterings)], cwd=self.clustering_program_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
		try:
			# while not os.path.exists(self.output_clustering_path + "results_1/tp"):
			# 	sleep(1)
				
			with open(self.output_clustering_path + "results_1/tp", "r") as read_file:
				with open(self.output_clustering_path + self.folder_name + ".dat", "w") as write_file:
					cluster_number_string = "1"
					for line in read_file:
						if line[0] == "#":
							continue
						nodes = line.split()
						for node in nodes:
							write_file.write(node + "\t" + cluster_number_string + "\n")
						cluster_number_string = str(int(cluster_number_string) + 1)
			# print("Successfully wrote " + self.output_clustering_path + self.folder_name + ".dat")
		except:
			print("Failed parsing results:", self.output_clustering_path + "results_1/tp")
	


class Leiden_clustering():
	"""
	Run clustering algorithm using Leiden code and store the results using the format (node_id>=0, membership>=1)
	@param: network_file_path (str). Path to the network file
	@param: output_clustering_path (str). The directory where the output is going to be placed
	@param: method (str). Community detection method to be used
	@return: None
	"""

	def __init__(self, network_file_path, output_clustering_path, method):
		self.network_file_path = network_file_path
		self.output_clustering_path = output_clustering_path
		self.method = method  

		# Hyperparameter
		self.clustering_program_path = os.path.dirname(os.getcwd()) + "/clustering_programs/"


	def __call__(self):
		"""
		Run clustering algorithm using Leiden code and store the results using the format (node_id>=0, membership>=0).
		Write clustering results in a file using the format (node_id>=0, memmbership>=1)
		"""  

		quality_function="modularity"
		resolution_parameter=1.0
		algorithm="Leiden"
		n_random_starts=10
		n_iterations=10

		process = subprocess.Popen(["time", "java", "-jar", "RunNetworkClustering.jar", "-q", quality_function, "-r", str(resolution_parameter), "-a", algorithm, "-s", str(n_random_starts), "-i", str(n_iterations), "-o", self.output_clustering_path, self.network_file_path], cwd=self.clustering_program_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = process.communicate()
		#print("error:", err)

		# subprocess.check_call(["time", "java", "-jar", "RunNetworkClustering.jar", "-q", quality_function, "-r", str(resolution_parameter), "-a", algorithm, "-s", str(n_random_starts), "-i", str(n_iterations), "-o", self.output_clustering_path, self.network_file_path], cwd=self.clustering_program_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
		try:
			# while not os.path.exists(self.output_clustering_path):
			# 	sleep(1)

			df = pd.read_csv(self.output_clustering_path, sep="\t", header=None)
			
			# Fix the membership range (+1)
			df.columns = ["id", "membership"]
			df["membership"] = df["membership"].to_numpy() + 1

			df.to_csv(self.output_clustering_path, index=False, header=False, sep="\t")
		except:
			print("Not found clustering file:", self.output_clustering_path)




### Modified from fastconsensus (GitHub of kaiser-dan)
class fast_consensus():

	def __init__(self, edge_list, np, alg, out_directory):
		self.edge_list = edge_list
		self.np = np
		self.alg = alg
		self.out_directory = out_directory

		## Hyperparameter tau and delta
		default_tau = {'louvain': 0.2, 'cnm': 0.7 ,'infomap': 0.6, 'lpm': 0.8, 'leiden': 0.2}
		# Default: default_tau = {'louvain': 0.2, 'cnm': 0.7 ,'infomap': 0.6, 'lpm': 0.8, 'leiden': 0.2}
		self.t = default_tau.get(self.alg)
		self.d = 0.02
		# Default: self.d = 0.02


	def __call__(self):

		if self.check_arguments() == False:

			quit()

		G = nx.Graph()
		G.add_edges_from(self.edge_list)
		
		output = self.fast_run(G)
		
		if not os.path.exists(self.out_directory):
			os.makedirs(self.out_directory[:-1])

		if(self.alg == 'louvain'):
			for i in range(len(output)):
				output[i] = self.group_to_partition(output[i])

		i = 0
		for partition in output:
			i += 1
		
			with open(self.out_directory + str(i) , 'w') as f:
				for community in partition:
					print(*community, file = f)


	def check_arguments(self):

		if(self.d > 0.2):
			print('delta is too high. Allowed values are between 0.02 and 0.2')
			return False
		if(self.d < 0.02):
			print('delta is too low. Allowed values are between 0.02 and 0.2')
			return False
		if(self.alg not in ('louvain', 'lpm', 'cnm', 'infomap', 'leiden')):
			print('Incorrect algorithm entered. run with -h for help')
			return False
		if (self.t > 1 or self.t < 0):
			print('Incorrect tau. run with -h for help')
			return False

		return True


	def check_consensus_graph(self, G):
		'''
		This function checks if the networkx graph has converged.
		Input:
		G: networkx graph
		np: number of partitions while creating G
		delta: if more than delta fraction of the edges have weight != n_p then returns False, else True
		'''

		count = 0

		for wt in nx.get_edge_attributes(G, 'weight').values():
			if wt != 0 and wt != self.np:
				count += 1

		if count > self.d*G.number_of_edges():
			return False

		return True


	def nx_to_igraph(self, Gnx):
		'''
		Function takes in a network Graph, Gnx and returns the equivalent
		igraph graph g
		'''
		g = ig.Graph()
		g.add_vertices(sorted(Gnx.nodes()))
		g.add_edges(sorted(Gnx.edges()))
		g.es["weight"] = 1.0
		for edge in Gnx.edges():
			g[edge[0], edge[1]] = Gnx[edge[0]][edge[1]]['weight']
		return g


	def group_to_partition(self, partition):
		'''
		Takes in a partition, dictionary in the format {node: community_membership}
		Returns a nested list of communities [[comm1], [comm2], ...... [comm_n]]
		'''

		part_dict = {}

		for index, value in partition.items():

			if value in part_dict:
				part_dict[value].append(index)
			else:
				part_dict[value] = [index]


		return part_dict.values()


	def communities_to_dict(self, communities):
		"""
		Creates a [node] -> [community] lookup
		"""
		result = {}
		community_index = 0
		for c in communities:
			community_mapping = ({str(node):community_index for _, node in enumerate(c)})

			result = {**result, **community_mapping}
			community_index += 1
		return result


	def edges_lookup_table_by_node(self, edges):
		"""
		Creates a [node] -> [[u,v]] lookup
		"""
		result = {}
		for u, v in edges:
			if u in result:
				result[u].append((u,v))
			else:
				result[u] = [(u,v)]

			if v in result:
				result[v].append((v,u))
			else:
				result[v] = [(v,u)]
		return result


	def do_leiden_community_detection(self, data):
		networkx_graph, seed = data
		return leidenalg.find_partition(self.nx_to_igraph(networkx_graph), leidenalg.ModularityVertexPartition, weights='weight',  seed=seed, n_iterations=1).as_cover()


	def get_graph_and_seed(self, graph):
		for seed in range(self.np):
			yield graph, seed


	def fast_run(self, G):

		graph = G.copy()
		L = G.number_of_edges()
		N = G.number_of_nodes()

		for u,v in graph.edges():
			graph[u][v]['weight'] = 1.0

		while(True):

			if (self.alg == 'louvain'):

				nextgraph = graph.copy()
				L = G.number_of_edges()
				for u,v in nextgraph.edges():
					nextgraph[u][v]['weight'] = 0.0

				communities_all = [cm.partition_at_level(cm.generate_dendrogram(graph, randomize = True, weight = 'weight'), 0) for i in range(self.np)]

				for node,nbr in graph.edges():

					if (node,nbr) in graph.edges() or (nbr, node) in graph.edges():
						if graph[node][nbr]['weight'] not in (0,self.np):
							for i in range(self.np):
								communities = communities_all[i]
								if communities[node] == communities[nbr]:
									nextgraph[node][nbr]['weight'] += 1
								else:
									nextgraph[node][nbr]['weight'] = graph[node][nbr]['weight']

				remove_edges = []
				for u,v in nextgraph.edges():
					if nextgraph[u][v]['weight'] < self.t*self.np:
						remove_edges.append((u, v))

				nextgraph.remove_edges_from(remove_edges)

				if self.check_consensus_graph(nextgraph):
					break

				for _ in range(L):

					node = np.random.choice(nextgraph.nodes())
					neighbors = [a[1] for a in nextgraph.edges(node)]

					if (len(neighbors) >= 2):
						a, b = random.sample(set(neighbors), 2)

						if not nextgraph.has_edge(a, b):
							nextgraph.add_edge(a, b, weight = 0)

							for i in range(self.np):
								communities = communities_all[i]

								if communities[a] == communities[b]:
									nextgraph[a][b]['weight'] += 1

				for node in nx.isolates(nextgraph):
						nbr, weight = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[0]
						nextgraph.add_edge(node, nbr, weight = weight['weight'])

				graph = nextgraph.copy()

				if self.check_consensus_graph(nextgraph):
					break

			elif self.alg == 'leiden':
				nextgraph = graph.copy()

				for u,v in nextgraph.edges():
					nextgraph[u][v]['weight'] = 0.0

				with mp.Pool(processes=self.np) as pool:
					communities = pool.map(self.do_leiden_community_detection, self.get_graph_and_seed(graph))

				for i in range(self.np):
					node_community_lookup = self.communities_to_dict(communities[i])
					for community_index, _ in enumerate(communities[i]):
						for node, nbr in graph.edges():
							if node in node_community_lookup and nbr in node_community_lookup and node_community_lookup[node] == node_community_lookup[nbr]:
								if node_community_lookup[node] != community_index:
									# only count each community once
									continue
								nextgraph[node][nbr]['weight'] += 1

				remove_edges = []
				for u,v in nextgraph.edges():
					if nextgraph[u][v]['weight'] < self.t*self.np:
						remove_edges.append((u, v))
				nextgraph.remove_edges_from(remove_edges)

				if self.check_consensus_graph(nextgraph):
					break

				for i in range(self.np):
					node_community_lookup = self.communities_to_dict(communities[i])
					n_graph_nodes = len(nextgraph.nodes())

					edges_lookup_table = self.edges_lookup_table_by_node(nextgraph.edges)

					for _ in range(L):
						random_node_index = random.randint(1, n_graph_nodes)
						neighbors = [a[1] for a in edges_lookup_table.get(str(random_node_index), [])]

						if (len(neighbors) >= 2):
							a, b = random.sample(set(neighbors), 2)

							if not nextgraph.has_edge(a, b):
								nextgraph.add_edge(a, b, weight = 0)

								if a in node_community_lookup and b in node_community_lookup and node_community_lookup[a] == node_community_lookup[b]:
									nextgraph[a][b]['weight'] += 1

				for node in nx.isolates(nextgraph):
					nbr, weight = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[0]
					nextgraph.add_edge(node, nbr, weight = weight['weight'])

				graph = nextgraph.copy()

				if self.check_consensus_graph(nextgraph):
					break

			elif (self.alg in ('infomap', 'lpm')):

				nextgraph = graph.copy()

				for u,v in nextgraph.edges():
					nextgraph[u][v]['weight'] = 0.0

				if self.alg == 'infomap':
					communities = [{frozenset(c) for c in self.nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(self.np)]
				if self.alg == 'lpm':
					communities = [{frozenset(c) for c in self.nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(self.np)]


				for node, nbr in graph.edges():

					for i in range(self.np):
						for c in communities[i]:
							if node in c and nbr in c:
								if not nextgraph.has_edge(node,nbr):
									nextgraph.add_edge(node, nbr, weight = 0)
								nextgraph[node][nbr]['weight'] += 1

				remove_edges = []
				for u,v in nextgraph.edges():
					if nextgraph[u][v]['weight'] < self.t*self.np:
						remove_edges.append((u, v))
				nextgraph.remove_edges_from(remove_edges)

				for _ in range(L):
					node = np.random.choice(nextgraph.nodes())
					neighbors = [a[1] for a in nextgraph.edges(node)]

					if (len(neighbors) >= 2):
						a, b = random.sample(set(neighbors), 2)

						if not nextgraph.has_edge(a, b):
							nextgraph.add_edge(a, b, weight = 0)

							for i in range(self.np):
								if a in communities[i] and b in communities[i]:
									nextgraph[a][b]['weight'] += 1

				graph = nextgraph.copy()

				if self.check_consensus_graph(nextgraph):
					break

			elif (self.alg == 'cnm'):

				nextgraph = graph.copy()

				for u,v in nextgraph.edges():
					nextgraph[u][v]['weight'] = 0.0

				communities = []
				mapping = []
				inv_map = []

				for _ in range(self.np):

					order = list(range(N))
					random.shuffle(order)
					maps = dict(zip(range(N), order))

					mapping.append(maps)
					inv_map.append({v: k for k, v in maps.items()})
					G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
					G_igraph = self.nx_to_igraph(G_c)

					communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())

				for i in range(self.np):

					edge_list = [(mapping[i][j], mapping[i][k]) for j,k in graph.edges()]

					for node,nbr in edge_list:
						a, b = inv_map[i][node], inv_map[i][nbr]

						if graph[a][b] not in (0, self.np):
							for c in communities[i]:
								if node in c and nbr in c:
									nextgraph[a][b]['weight'] += 1

						else:
							nextgraph[a][b]['weight'] = graph[a][b]['weight']

				remove_edges = []
				for u,v in nextgraph.edges():
					if nextgraph[u][v]['weight'] < self.t*self.np:
						remove_edges.append((u, v))

				nextgraph.remove_edges_from(remove_edges)

				for _ in range(L):
					node = np.random.choice(nextgraph.nodes())
					neighbors = [a[1] for a in nextgraph.edges(node)]

					if (len(neighbors) >= 2):
						a, b = random.sample(set(neighbors), 2)
						if not nextgraph.has_edge(a, b):
							nextgraph.add_edge(a, b, weight = 0)

							for i in range(self.np):
								for c in communities[i]:
									if mapping[i][a] in c and mapping[i][b] in c:

										nextgraph[a][b]['weight'] += 1

				if self.check_consensus_graph(nextgraph):
					break

			else:
				break

		if (self.alg == 'louvain'):
			return [cm.partition_at_level(cm.generate_dendrogram(graph, randomize = True, weight = 'weight'), 0) for _ in range(self.np)]
		if self.alg == 'leiden':
			with mp.Pool(processes=self.np) as pool:
				communities = pool.map(self.do_leiden_community_detection, self.get_graph_and_seed(graph))
			return communities
		if self.alg == 'infomap':
			return [{frozenset(c) for c in self.nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(self.np)]
		if self.alg == 'lpm':
			return [{frozenset(c) for c in self.nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(self.np)]
		if self.alg == 'cnm':

			communities = []
			mapping = []
			inv_map = []

			for _ in range(self.np):
				order = list(range(N))
				random.shuffle(order)
				maps = dict(zip(range(N), order))

				mapping.append(maps)
				inv_map.append({v: k for k, v in maps.items()})
				G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
				G_igraph = self.nx_to_igraph(G_c)

				communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())

			return communities

