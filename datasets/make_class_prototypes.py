import json
from networkx import normalized_laplacian_matrix
import sys
import os
import scipy
import numpy as np
import glob
import ot
import networkx as nx

def create_prototypes(dataset_name):
	all_files = glob.glob(dataset_name + "/json_format/*")
	all_files.sort(key=lambda x: int((x.strip().split('/')[-1]).split('.')[0]))

	all_clases = {}

	for file in all_files:
		name = (file.strip().split('/')[-1]).split('.')[0]
		with open(file, "r") as f1:
			graph = json.load(f1)

		if graph["target"] not in all_clases.keys():
			all_clases[graph["target"]] = {}

		if len(graph["labels"]) == 1:
			raise ValueError("Only one node")

		G = nx.Graph()
		G.add_edges_from(graph["edges"])
		N = normalized_laplacian_matrix(G).todense()
		eigvals = scipy.linalg.eigvals(N)
		eigvals = eigvals.real.round(decimals=5)
		if type(eigvals) == int:
			raise TypeError("Type is int rather than list")

		all_clases[graph["target"]][int(name)] = eigvals

	class_prototype_dict = {}

	for class_, class_graphs in all_clases.items():
		current_class_eigvals = []

		for num in sorted(class_graphs):
			# print(num)
			current_class_eigvals.append(class_graphs[num])

		all_dist = []
		for i in range(len(current_class_eigvals)):
			current_dist = []
			for j in range(len(current_class_eigvals)):
				a = current_class_eigvals[i]
				b = current_class_eigvals[j]
				cost = ot.utils.dist( np.reshape(a, (a.shape[0], 1)), np.reshape(b, (b.shape[0], 1)) )
				loss = ot.emd2([], [], cost)
				current_dist.append(loss)
			all_dist.append(current_dist)
		all_dist = np.array(all_dist)
		current_prot_index = np.argmin(np.sum(all_dist, axis=1))
		# print(list(class_graphs.keys()))
		sorted_keys = list(class_graphs.keys())
		sorted_keys.sort()
		class_prototype_dict[str(class_)] = sorted_keys[current_prot_index]

	print(class_prototype_dict)
	with open(dataset_name + "/class_prototype_numbers.json", 'w') as f:
		json.dump(class_prototype_dict, f)

if __name__ == '__main__':
	dataset_name = sys.argv[1]
	create_prototypes(dataset_name)
	pass