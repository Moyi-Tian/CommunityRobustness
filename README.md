# CommunityRobustness
Studying robustness of community partitions in synthetic and empirical networks under edge addition using community detection algorithms



## Requirement
Our computation is launched on Ubuntu 20.04.5 LTS (GNU/Linux 5.4.0-135-generic x86_64) with 40 cores.

All dependencies for running our experiments are provided in `env.yml`. You can set up a conda environment using commands like the following:  

```
conda env create -n robustness --file env.yml
conda activate robustness
```



## Reproducing Experiments

Both the synthetic experiments using the Lancichinetti-Fortunato-Radicch (LFR) benchmark graphs and the empirical experiments can be reproduced with

`python driver.py`

Then follow the popped up inquiries to provide user inputs correspondingly. Hit `enter` each time to proceed. The algorithm will run automatically after all inputs required are provided appropriately.



The results are written into `json` data files. If running synthetic experiments on LFR benchmark, the result files will be in `./Data/LFR_network` and the file names will start with `results_`. When running on empirical datasets, information on the initial partition and computational results over timesteps are recorded in separate folders, which can be found in the directory `./Data/empirical_network/`. These corresonding directories and files are generated automatically by the algorithm.



### Datasets

We provide 3 empirical network datasets in `./Empirical_Dataset` as examples to run empirical experiments. These data are pre-processed and cleaned from the following original datasets obtained from online data repositories:

1. R. Rossi and N. Ahmed, Ia-radoslaw-email, https://networkrepository.com/ia-radoslaw-email.php
2. J. Kunegis, Enron, http://konect.cc/networks/enron
3. J. Leskovec, email-eu-core temporal network, http://snap.stanford.edu/data/email-Eu-core-temporal.html





### References

Our code uses the following packages/algorithms developed by the others:



#### * LFR benchmark graph 

```
@article{PhysRevE.78.046110,
  title = {Benchmark graphs for testing community detection algorithms},
  author = {Lancichinetti, Andrea and Fortunato, Santo and Radicchi, Filippo},
  journal = {Phys. Rev. E},
  volume = {78},
  issue = {4},
  pages = {046110},
  numpages = {5},
  year = {2008},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.78.046110},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.78.046110}
}
```

The package is provided on the author's webpage: https://www.santofortunato.net/resources



#### * Lancichinetti's program as a collection of clustering algorithms (specifically used Louvain, Infomap and Label Propagation)

The program itself is an implementation of the paper "Directed, weighted and overlapping benchmark graphs for community detection algorithms", written by Andrea Lancichinetti and Santo Fortunato. 

The package is provided on Santo Fortunato's Webpage https://www.santofortunato.net/resources where you can find the corresponding download link https://drive.google.com/file/d/1Z_ksrI559cSp6db3hrKhe1BmnyHGuIxa/view



#### * Leiden algorithm

```
@article{Traag_2019,
	doi = {10.1038/s41598-019-41695-z},
	url = {https://doi.org/10.1038%2Fs41598-019-41695-z},
	year = {2019},
	month = {mar}, 
	publisher = {Springer Science and Business Media {LLC}},
	volume = {9},
	number = {1},
	author = {V. A. Traag and L. Waltman and N. J. van Eck},	  
	title = {From Louvain to Leiden: guaranteeing well-connected communities},	  
	journal = {Scientific Reports}
}
```

Leiden package GitHub page: https://github.com/CWTSLeiden/networkanalysis



#### * Fast Consensus

```
@article{PhysRevE.99.042301,
  title = {Fast consensus clustering in complex networks},
  author = {Tandon, Aditya and Albeshri, Aiiad and Thayananthan, Vijey and Alhalabi, Wadee and Fortunato, Santo},
  journal = {Phys. Rev. E},
  volume = {99},
  issue = {4},
  pages = {042301},
  numpages = {5},
  year = {2019},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.99.042301},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.99.042301}
}
```

Paper's corresponding GitHub page: https://github.com/kaiser-dan/fastconsensus
