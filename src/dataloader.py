import numpy as np
import torch
import networkx as nx
import os
from torch_geometric.nn import knn_graph
import sys
from util import load_data
import math
import random
import json
from networkx import normalized_laplacian_matrix
from sklearn.neighbors import KNeighborsClassifier
from grakel import GraphKernel
import scipy
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import ot

class Dataset:
	def __init__(self, name, args):
		self.dataset_name = name
		self.num_clusters = {"train": args.train_clusters}
		print("\nNumber of super clusters = ", args.train_clusters, "\n")

		self.train_graphs = []
		self.test_graphs = []

	def segregate(self, args, model_run):
		all_graphs, label_dict = load_data(self.dataset_name, True)
		num_classes = len(label_dict)

		all_classes = list(label_dict.keys())

		if 0 in all_classes:
			class_label_start_from_zero = True
		else:
			class_label_start_from_zero = False

		print("Loading class splits")
		with open("./checkpoints/{}/main_splits.json".format(args.dataset_name), "r") as f:
			all_class_splits = json.load(f)
			self.train_classes = all_class_splits["train"]
			self.test_classes = all_class_splits["test"]

		train_classes_mapping = {}
		for cl in self.train_classes:
			train_classes_mapping[cl] = len(train_classes_mapping)
		test_classes_mapping = {}
		for cl in self.test_classes:
			test_classes_mapping[cl] = len(test_classes_mapping)

		for i in range(len(all_graphs)):
			if all_graphs[i].label in self.train_classes:
				self.train_graphs.append(all_graphs[i])

			if all_graphs[i].label in self.test_classes:
				self.test_graphs.append(all_graphs[i])

		with open("../datasets/" + self.dataset_name + "/class_prototype_numbers.json", "r") as f:
			prototypes = json.load(f)

		train_prototypes = {}
		for key, value in prototypes.items():
			if int(key) in self.train_classes:
				train_prototypes[int(key)] = value
			
		self.train_super_classes = self.get_super_classes(train_prototypes, "train", args)
		for graph in self.train_graphs:
			graph.super_class = self.train_super_classes[int(graph.label)]
			graph.label = train_classes_mapping[int(graph.label)]

		num_validation_graphs = math.floor(0.2 * len(self.train_graphs))
		self.train_graphs = self.train_graphs[ : len(self.train_graphs) - num_validation_graphs]
		self.validation_graphs = self.train_graphs[len(self.train_graphs) - num_validation_graphs : ]

		print("Number of training graphs = ", len(self.train_graphs))
		print("Number of validation graphs = ", len(self.validation_graphs))
		print("Number of total testing graphs = ", len(self.test_graphs), "\n")

		self.split_test_set_for_fine_tuning(args.n_shot)
		for i, graph in enumerate(self.test_fine_tuning_graphs):
			graph.label = test_classes_mapping[int(graph.label)]

		for i, graph in enumerate(self.final_testing_graphs):
			graph.label = test_classes_mapping[int(graph.label)]


	def get_super_classes(self, prototype_json, type_, args):
		prototypes_list = []
		for class_ in sorted(prototype_json):
			name = str(prototype_json[class_])
			with open("../datasets/" + self.dataset_name + "/json_format/{}.json".format(name), "r") as f1:
				graph = json.load(f1)

			G = nx.Graph()
			G.add_edges_from(graph["edges"])
			N = normalized_laplacian_matrix(G).todense()
			eigvals = scipy.linalg.eigvals(N)
			eigvals = eigvals.real.round(decimals=5)
			if type(eigvals) == int:
				raise TypeError("Type is int rather than list")

			prototypes_list.append(eigvals)

		all_dist = []
		for i in range(len(prototypes_list)):
			current_dist = []
			for j in range(len(prototypes_list)):
				a = prototypes_list[i]
				b = prototypes_list[j]
				cost = ot.utils.dist( np.reshape(a, (a.shape[0], 1)), np.reshape(b, (b.shape[0], 1)) )
				loss = ot.emd2([], [], cost)
				current_dist.append(loss)
			all_dist.append(current_dist)
		kernel_matrix = np.array(all_dist)

		# This is used as an approximation with the Lloyd's variant proposed in the paper. Well intergrated with the scikit-learn
		# library, its assures better implementation and was thus used in the final version.
		clustering_super_class = AgglomerativeClustering(n_clusters=self.num_clusters["train"], 
					affinity="precomputed", linkage="complete").fit(kernel_matrix)

		super_class_dict = {}
		class_2_super_dict = {}
		
		classes = list(prototype_json.keys())
		classes.sort()
		super_class_labels = list(clustering_super_class.labels_)

		for i in range(len(super_class_labels)):
			if super_class_labels[i] not in super_class_dict.keys():
				super_class_dict[super_class_labels[i]] = []
			super_class_dict[super_class_labels[i]].append(classes[i])
			class_2_super_dict[classes[i]] = super_class_labels[i]

		return class_2_super_dict


	def split_test_set_for_fine_tuning(self, n_shot):
		self.test_fine_tuning_graphs = []
		self.final_testing_graphs = []

		random.shuffle(self.test_graphs)
		test_n_shot = {}

		for i in range(len(self.test_graphs)):
			if self.test_graphs[i].label not in test_n_shot.keys():
				test_n_shot[self.test_graphs[i].label] = []
			if len(test_n_shot[self.test_graphs[i].label]) < n_shot:
				test_n_shot[self.test_graphs[i].label].append(self.test_graphs[i])
			else:
				self.final_testing_graphs.append(self.test_graphs[i])

		for key, val in test_n_shot.items():
			self.test_fine_tuning_graphs += val
		print("\nNumber of test fine tuning graphs = ", len(self.test_fine_tuning_graphs))
		print("Number of test evaluation graphs = ", len(self.final_testing_graphs), "\n")

	def create_train_knn_graph(self, embeds, batch, args):
		super_class_segregation = {}
		actual_ranking = {}
		count = 0
		
		for i, graph in enumerate(batch):
			if graph.super_class not in super_class_segregation.keys():
				super_class_segregation[graph.super_class] = []
			if graph.super_class not in actual_ranking.keys():
				actual_ranking[graph.super_class] = {}
			super_class_segregation[graph.super_class].append(embeds[i].unsqueeze(0))
			actual_ranking[graph.super_class][len(actual_ranking[graph.super_class])] = i

		all_edges = []

		for key, value in super_class_segregation.items():
			knn_value = args.knn_value

			super_class_embeds = torch.cat(value, dim=0)
			super_class_knn = knn_graph(super_class_embeds, knn_value, loop=True)

			actual_super_class_knn = np.zeros((super_class_knn.shape[0], super_class_knn.shape[1])).astype(np.int32)
			for i in range(super_class_knn.shape[0]):
				for j in range(super_class_knn.shape[1]):
					actual_super_class_knn[i, j] = actual_ranking[key][int(super_class_knn[i, j].cpu().numpy())]

			all_edges.append(torch.LongTensor(actual_super_class_knn).cuda())

		return torch.cat(all_edges, dim=1)

	def create_test_knn_graph(self, embeds, batch, args, gin_preds):
		super_class_segregation = {}
		actual_ranking = {}
		count = 0
		super_class_preds = torch.argmax(gin_preds, dim=1).cpu().numpy()
		
		for i, graph in enumerate(batch):
			if super_class_preds[i] not in super_class_segregation.keys():
				super_class_segregation[super_class_preds[i]] = []
			if super_class_preds[i] not in actual_ranking.keys():
				actual_ranking[super_class_preds[i]] = {}
			super_class_segregation[super_class_preds[i]].append(embeds[i].unsqueeze(0))
			actual_ranking[super_class_preds[i]][len(actual_ranking[super_class_preds[i]])] = i

		all_edges = []

		for key, value in super_class_segregation.items():
			knn_value = args.knn_value

			super_class_embeds = torch.cat(value, dim=0)
			super_class_knn = knn_graph(super_class_embeds, knn_value, loop=True)

			actual_super_class_knn = np.zeros((super_class_knn.shape[0], super_class_knn.shape[1])).astype(np.int32)
			for i in range(super_class_knn.shape[0]):
				for j in range(super_class_knn.shape[1]):
					actual_super_class_knn[i, j] = actual_ranking[key][int(super_class_knn[i, j].cpu().numpy())]

			all_edges.append(torch.LongTensor(actual_super_class_knn).cuda())

		return torch.cat(all_edges, dim=1)