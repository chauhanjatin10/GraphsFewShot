import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time
import numpy as np
import json
import random

from model import FewShotModel
from dataloader import Dataset
from util import save_test_embeddings

def parse_arguments():
	parser = argparse.ArgumentParser()
	
	# GIN parameters
	parser.add_argument('--dataset_name', type=str, default="TRIANGLES",
						help='name of dataset')
	parser.add_argument('--device', type=int, default=0,
						help='which gpu to use if any (default: 0)')
	parser.add_argument('--num_layers', type=int, default=5,
						help='number of layers INCLUDING the input one (default: 5)')
	parser.add_argument('--num_mlp_layers', type=int, default=2,
						help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
	parser.add_argument('--hidden_dim', type=int, default=128,
						help='number of hidden units (default: 64)')
	parser.add_argument('--final_dropout', type=float, default=0.5,
						help='final layer dropout (default: 0.5)')
	parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
						help='Pooling for over nodes in a graph: sum or average')
	parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
						help='Pooling for over neighboring nodes: sum, average or max')
	parser.add_argument('--learn_eps', action="store_true",
										help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
	parser.add_argument('--degree_as_tag', action="store_true",
						help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
	
	# GAT parameters
	parser.add_argument('--num_gat_layers', type=int, default=2, help="number of GAT layers")
	parser.add_argument('--gat_out_dim', type=int, default=128, help="GAT output dims")
	parser.add_argument('--gat_dropout', type=float, default=0.5, help="GAT dropout")
	parser.add_argument('--gat_heads', type=int, default=2, help="number of head per layer")
	parser.add_argument('--gat_leaky_slope', type=float, default=0.1, help="slope of leaky relu")
	parser.add_argument('--gat_concat', type=int, default=0, help="whether to concat embeddings of\
										the GAT layer heads")

	# Main model parameters
	parser.add_argument('--is_train', type=int, default=1, help='whether to train the model or directly test')
	parser.add_argument('--is_test', type=int, default=1, help='whether test after training the model')

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--iters_per_epoch', type=int, default=10)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=1e-7)
	parser.add_argument('--save_model_after', type=int, default=1,
						help='saving after this many epochs')

	
	# Fine Tuning Params
	parser.add_argument('--n_shot', type=int, default=20)
	parser.add_argument('--model_runs', type=int, default=50)
	parser.add_argument('--knn_value', type=int, default=2)
	parser.add_argument('--train_clusters', type=int, default=3)
	parser.add_argument('--num_inference_graphs', type=int, default=500)
	
	parser.add_argument('--fine_tune_lr', type=float, default=0.001)
	parser.add_argument('--fine_tune_weight_decay', type=float, default=1e-7)
	parser.add_argument('--fine_tune_epochs', type=int, default=20)
	parser.add_argument('--fine_tune_iters_per_epoch', type=int, default=10)
	parser.add_argument('--fine_tune_batch_size', type=int, default=500)

	parser.add_argument('--fine_tune_save_model_after', type=int, default=1)
	parser.add_argument('--num_testing_runs', type=int, default=10)

	args = parser.parse_args()
	return args

args = parse_arguments()

def run_model():
	knn_params = {"initial": args.knn_value}
	gat_layer_params = {}

	for i in range(args.num_gat_layers):
		knn_params[i] = args.knn_value

		gat_layer_params[i] = {}
		if i == 0:
			gat_layer_params[i]["in_channels"] = args.hidden_dim * (args.num_layers-1)
			gat_layer_params[i]["out_channels"] = args.gat_out_dim

		else:
			if args.gat_concat == 1:
				gat_layer_params[i]["in_channels"] = gat_layer_params[i-1]["out_channels"] * \
													 gat_layer_params[i-1]["heads"]
			else:
				gat_layer_params[i]["in_channels"] = gat_layer_params[i-1]["out_channels"]
			gat_layer_params[i]["out_channels"] = args.gat_out_dim
		
		if args.gat_concat == 1:
			gat_layer_params[i]["concat"] = True
		else:
			gat_layer_params[i]["concat"] = False
		gat_layer_params[i]["heads"] = args.gat_heads
		gat_layer_params[i]["leaky_slope"] = args.gat_leaky_slope
		gat_layer_params[i]["dropout"] = args.gat_dropout

	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	print(args, "\n")

	overall_acc = []
	for runs in range(args.model_runs):
		print("\nCurrent run = ", runs+1, "\n")
		main_start_time = time.time()

		dataset = Dataset(args.dataset_name, args)
		dataset.segregate(args, runs)

		if args.is_train == 1:
			train_start_time = time.time()
			
			train_start_time = time.time()
			few_shot_model = FewShotModel(args.num_layers, args.num_mlp_layers, 
					dataset.train_graphs[0].node_features.shape[1], args.hidden_dim, 
					args.train_clusters,
					args.final_dropout, args.learn_eps, args.graph_pooling_type, 
					args.neighbor_pooling_type, device, args.num_gat_layers, gat_layer_params, 
					knn_params, dataset, "train").to(device)

			few_shot_model.train_(args, dataset)
			print("\nRemaining Training phase finished in ", time.time() - train_start_time, " seconds\n")

		if args.is_test == 1:
			print("\nFine tuning GIN for super-class classification on test classes...\n")
			test_start_time = time.time()
			
			few_shot_model = FewShotModel(args.num_layers, args.num_mlp_layers, 
					dataset.train_graphs[0].node_features.shape[1], args.hidden_dim, 
					args.train_clusters,
					args.final_dropout, args.learn_eps, args.graph_pooling_type, 
					args.neighbor_pooling_type, device, args.num_gat_layers, gat_layer_params, 
					knn_params, dataset, "test").to(device)

			few_shot_model.test_fine_tuning(args, dataset, device)
			print("\nFine tuning finished in ", time.time() - test_start_time, " seconds\n")
			current_run_acc = few_shot_model.test(args, dataset, device)
			overall_acc.append(current_run_acc)

		print("\nTotal time taken for 1 complete run = ", time.time() - main_start_time, " seconds\n")
	print("\nAverage acc over 50 runs = ", np.mean(overall_acc), " , std = ", np.std(overall_acc))

if __name__ == '__main__':
	run_model()