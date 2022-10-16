import os
import shutil
import errno
import subprocess
import random
from time import sleep



class LFR_Benchmark():
    def __init__(self, program_path, N, k, maxk, mu, minc, maxc, out_dir_path):
        """
        @param: program_path (str). The path to the LFR executable
        @param: N (int). Number of nodes 
        @param: k (int). Average degree
        @param: maxk (int). Maximum degree 
        @param: mu (float). Mixing parameter
        @param: minc (int). Minimun community size
        @param: maxc (int). Maximum community size
        @param: out_dir_path (str). The path where the output folder is created
        """
        self.program_path = program_path
        self.N = N
        self.k = k
        self.maxk = maxk
        self.mu = mu
        self.minc = minc
        self.maxc = maxc
        self.out_dir_path = out_dir_path
    

    def __call__(self):
        """
        Run a LFR realization and prune the output files. Output file are placed in the out_dir_path
        """

        # Flags file name
        flag_file_name = "myflags.dat"

        # Create flag file to generate LFR
        self.generate_flag_file(flag_file_name)
        # Create output folder
        self.create_folder_if_needed()

        # Call LFR C++ program
        subprocess.call(["./benchmark", "-f", flag_file_name], cwd=self.program_path)

        # sleep(random.randint(10, 100))

        ## Move community detection output files to the output directory
        # network.dat contains the list of edges (nodes are labelled from 1 to the number of nodes; the edges are ordered and repeated twice, i.e. source-target and target-source).
        # print("LFR data file exists? : ", os.path.exists(self.program_path + "network.dat"))
        common_name = "_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + "_maxk_" + str(self.maxk) + "_minc_" + str(self.minc) + "_maxc_" + str(self.maxc) + ".dat"
        shutil.move(self.program_path + "network.dat", self.out_dir_path + "network" + common_name)
        
        # community.dat contains a list of the nodes and their membership (memberships are labelled by integer numbers >=1).
        shutil.move(self.program_path + "community.dat", self.out_dir_path + "community" + common_name)
        
        # statistics.dat contains the degree distribution (in logarithmic bins), the community size distribution, and the distribution of the mixing parameter.
        shutil.move(self.program_path + "statistics.dat", self.out_dir_path + "statistics" + common_name)
        
        # Remove duplicate edges from edgelist file and rewrite edgelist file such that node ids start from zero for compatibility with clustering program input formats
        self.remove_duplicate_edges(self.out_dir_path + "network" + common_name, assume_one_max=True)
        self.rewrite_edgelist_from_zero(self.out_dir_path + "network" + common_name)
            
        # Rewrite clustering file such that node ids start from zero to maintain consistency with edgelist file node ids
        self.rewrite_clustering_from_zero(self.out_dir_path + "community" + common_name)


    
    def generate_flag_file(self, file_name):
        """
        Generates the text file that specify argument values to LFR
        @param: file_name (str). The name of the file (myflags.dat)
        @return: None
        """

        to_write = ""

        to_write += "-N " + str(self.N) + "\n"
        to_write += "-k " + str(self.k) + "\n"
        to_write += "-maxk " + str(self.maxk) + "\n"
        to_write += "-mu " + str(self.mu) + "\n"
        to_write += "-t1 2\n"
        to_write += "-t2 1\n"
        to_write += "-minc " + str(self.minc) + "\n"
        to_write += "-maxc " + str(self.maxc) + "\n"
        to_write += "-on 0\n"
        to_write += "-om 0\n"

        with open(self.program_path + file_name, "w") as f:
            f.write(to_write)


    def create_folder_if_needed(self):
        """
        Creates a non existing folder
        @return: None
        """

        try:
            os.makedirs(self.out_dir_path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise


    def remove_duplicate_edges(self, out_file_name, assume_one_max=False):
        """
        Given a network file removes edges such that the final network out_file_name
        contains no two edges that connect the same pair of nodes. Assumes node ids and cluster ids are integers.
        @param: out_file_name (str). Output network filename
        @param: assume_one_max (boolean). If assume_one_max, the function will assume that there are at most two edges in the original file connecting the same pair of nodes.
        @return: None
        """

        read_file = out_file_name
        write_file = "temporary_file_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + ".dat"
        #assert not os.path.isfile(write_file)

        separator = "\t"

        with open(read_file, "r") as read_f:
            with open(write_file, "w") as write_f:
                redundant_edges = {}
                empty_set = set() 
                for line in read_f:
                    source, destination = line.split(separator)
                    source = int(source)
                    destination = int(destination.rstrip()) # remove newline character and trailing spaces
                    if not destination in redundant_edges.get(source, empty_set):
                        write_f.write(str(source) + separator + str(destination) + "\n")
                        redundant_edges[destination] = redundant_edges.get(destination, empty_set)
                        redundant_edges[destination].add(source)
                        empty_set = set() # reverse mutation due to previous line
                    elif assume_one_max:
                        redundant_edges[source].remove(destination)
        
        # Move the temporary file to the destination file
        shutil.move(write_file, read_file)


    def get_min_edgelist_id(self, edgelist_file_name):
        """
        Obtain the minimum node id in an edgelist
        @param: edgelist_file_name (str). Input edgelist file
        @return: min_id (int). Lowest id in the edgelist
        """

        separator = "\t"

        with open(edgelist_file_name, "r") as f:
            source_id, destination_id = f.readline().split(separator) # read the first line of a file
            destination_id = destination_id[:-1] # remove newline character
            min_id = min(int(source_id), int(destination_id))
            for line in f:
                source_id, destination_id = line[:-1].split(separator) # line[:-1] removes newline character from destination_id 
                min_id = min(int(source_id), int(destination_id), min_id)

        return min_id



    def rewrite_edgelist_from_zero(self, graph_file_name):
        """
        Rewrite edgelist starting from index 0. This is to make edgelist compatible with most network libraries
        @param: graph_file_name (str). Output network filename
        @return: None
        """

        temporary_file = "temporary_file_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + ".dat"
        #assert not os.path.isfile(temporary_file)

        separator = "\t"
        min_id = self.get_min_edgelist_id(graph_file_name)

        with open(graph_file_name, "r") as source:
            with open(temporary_file, "w") as destination:
                for line in source:
                    source_id, destination_id = line[:-1].split(separator) # line[:-1] removes newline character from destination_id
                    source_id = str(int(source_id) - min_id)
                    destination_id = str(int(destination_id) - min_id)
                    destination.write(source_id + separator + destination_id + "\n")

        # Move the temporary file to the destination file
        shutil.move(temporary_file, graph_file_name)


    def get_min_clustering_id(self, clustering_file_name):
        """
        Obtain the minimum node id in the clustering file
        @param: clustering_file_name (str). Input clustering file
        @return: min_id (int). Lowest id in the clustering file
        """

        separator = "\t"
        
        with open(clustering_file_name, "r") as f:
            min_id = int(f.readline().split(separator)[0])
            for line in f:
                node_id = int(line.split(separator)[0])
                min_id = min(node_id, min_id)

        return min_id


    def rewrite_clustering_from_zero(self, clustering_file_name):
        """
        Rewrite clustering file such that node ids start from zero to maintain consistency with edgelist file node ids
        @param: clustering_file_name (str). Output network filename
        @return: None
        """

        temporary_file = "temporary_file_N_" + str(self.N) + "_mu_" + str(self.mu) + "_k_" + str(self.k) + ".dat"
        #assert not os.path.isfile(temporary_file)

        separator = "\t"
        min_id = self.get_min_clustering_id(clustering_file_name)

        with open(clustering_file_name, "r") as source:
            with open(temporary_file, "w") as destination:
                for line in source:
                    node_id, cluster_id = line[:-1].split(separator) #line[:-1] removes newline character from cluster_id
                    node_id = str(int(node_id) - min_id)
                    destination.write(node_id + separator + cluster_id + "\n")

        # Move the temporary file to the destination file
        shutil.move(temporary_file, clustering_file_name)
