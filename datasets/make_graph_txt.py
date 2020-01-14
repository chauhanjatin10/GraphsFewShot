import json
import sys
import os
import glob
import networkx as nx

dataset_name = sys.argv[1]

def create_txt():
	all_files = glob.glob(dataset_name + "/json_format/*")
	all_files.sort(key=lambda x: int((x.strip().split('/')[-1]).split('.')[0]))

	with open(dataset_name + "/{}.txt".format(dataset_name), "w") as f:
		f.write(str(len(all_files)) + "\n")
		
		for file in all_files:
			with open(file, "r") as f1:
				graph = json.load(f1)
			G = nx.Graph()
			G.add_edges_from(graph["edges"])

			f.write(str(len(graph["labels"])) + " " + str(graph["target"]) + "\n")

			for i in range(len(graph["labels"])):
				neigh = G.neighbors(i)
				neigh = [ int(nei) for nei in neigh ]
				f.write(graph["labels"][str(i)] + " " + str(len(neigh)) + " " )
				for j in neigh:
					f.write(str(j) + " ")
				f.write("\n")

if __name__ == '__main__':
	create_txt()
	pass