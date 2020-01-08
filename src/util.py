import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import ot
import sys
import scipy
import sklearn
import json

class S2VGraph(object):
	def __init__(self, g, label, node_tags=None, node_features=None):
		'''
			g: a networkx graph
			label: an integer graph label
			node_tags: a list of integer node tags
			node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
			edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
			neighbors: list of neighbors (without self-loop)
		'''
		self.label = label
		self.super_class = None
		self.g = g
		self.node_tags = node_tags
		self.neighbors = []
		self.node_features = 0
		self.edge_mat = 0

		self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
	'''
		dataset: name of dataset
		test_proportion: ratio of test train split
		seed: random seed for random splitting of dataset
	'''

	print('loading data')
	g_list = []
	label_dict = {}
	feat_dict = {}

	with open('../datasets/%s/%s.txt' % (dataset, dataset), 'r') as f:
		n_g = int(f.readline().strip())
		for i in range(n_g):
			row = f.readline().strip().split()
			n, l = [int(w) for w in row]
			if not l in label_dict:
				mapped = len(label_dict)
				label_dict[l] = mapped
			g = nx.Graph()
			node_tags = []
			node_features = []
			n_edges = 0
			for j in range(n):
				g.add_node(j)
				row = f.readline().strip().split()
				tmp = int(row[1]) + 2
				if tmp == len(row):
					# no node attributes
					row = [int(w) for w in row]
					attr = None
				else:
					row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
				if not row[0] in feat_dict:
					mapped = len(feat_dict)
					feat_dict[row[0]] = mapped
				node_tags.append(feat_dict[row[0]])

				if tmp > len(row):
					node_features.append(attr)

				n_edges += row[1]
				for k in range(2, len(row)):
					g.add_edge(j, row[k])

			if node_features != []:
				node_features = np.stack(node_features)
				node_feature_flag = True
			else:
				node_features = None
				node_feature_flag = False

			assert len(g) == n

			g_list.append(S2VGraph(g, l, node_tags))

	#add labels and edge_mat
	for g in g_list:
		g.neighbors = [[] for i in range(len(g.g))]
		for i, j in g.g.edges():
			g.neighbors[i].append(j)
			g.neighbors[j].append(i)
		degree_list = []
		for i in range(len(g.g)):
			g.neighbors[i] = g.neighbors[i]
			degree_list.append(len(g.neighbors[i]))
		g.max_neighbor = max(degree_list)

		# g.label = label_dict[g.label]

		edges = [list(pair) for pair in g.g.edges()]
		edges.extend([[i, j] for j, i in edges])

		deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
		g.edge_mat = torch.LongTensor(edges).transpose(0,1)

	if degree_as_tag:
		for g in g_list:
			g.node_tags = list(dict(g.g.degree).values())

	#Extracting unique tag labels
	tagset = set([])
	for g in g_list:
		tagset = tagset.union(set(g.node_tags))

	tagset = list(tagset)
	tag2index = {tagset[i]:i for i in range(len(tagset))}

	for g in g_list:
		g.node_features = torch.zeros(len(g.node_tags), len(tagset))
		g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


	print('# classes: %d' % len(label_dict))
	print('# maximum node tag: %d' % len(tagset))

	print("# data: %d" % len(g_list), "\n")

	return g_list, label_dict

def plot_tsne(dataset_name, labels, n_shot, embeds, run):
	plt.clf()
	tsne_embeds = TSNE(n_components=2).fit_transform(embeds)
	color_scheme = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "violet", 5: "purple"}
	embed_colors = [color_scheme[label] for label in labels]
	plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], color=embed_colors)
	plt.axis('off')
	plt.savefig('./tsne_plots/{0}/{1}shot/tsne_{2}.png'.format(dataset_name, n_shot, run))

def save_test_embeddings(embeds, labels, dataset_name, n_shot, model_run):
	np_labels = np.array(labels).reshape(-1, 1)
	labels_with_embeds = np.hstack( (np_labels, embeds) )
	np.save("./labels_with_embeds/{0}/{1}shot/labels_with_embeds.npy".format(dataset_name, n_shot), labels_with_embeds)
	plot_tsne(dataset_name, labels, n_shot, embeds, model_run)

def get_silhoutte(dataset_name, n_shot):
	labels_with_embeds = np.load("./labels_with_embeds/{0}/{1}shot/labels_with_embeds.npy".format(dataset_name, n_shot))
	labels = labels_with_embeds[:, 0]
	embeds = labels_with_embeds[:, 1:]
	embeds = embeds/np.linalg.norm(embeds, ord=2, axis=1, keepdims=True)
	dist = ot.utils.dist(embeds, embeds)
	score = sklearn.metrics.silhouette_score(dist, labels)
	print("silhouette_score = ", score)

if __name__ == '__main__':
	dataset_name = sys.argv[1]
	n_shot = sys.argv[2]
	# get_silhoutte(dataset_name, n_shot)