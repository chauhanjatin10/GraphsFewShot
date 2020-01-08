import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from graphcnn import GraphCNN, GINClassifier
from torch_geometric.nn import GATConv, knn_graph, GCNConv
import numpy as np
import random
from copy import deepcopy
import time
import math


class GATLayer(nn.Module):
	def __init__(self, gat_params, knn_value):
		super().__init__()
		self.knn_value = knn_value
		in_channels = gat_params["in_channels"]
		out_channels = gat_params["out_channels"]
		num_heads = gat_params["heads"]
		concat = gat_params["concat"] 	# bool value - whether to concat the multi-head embeddings
		leaky_slope = gat_params["leaky_slope"]
		dropout = gat_params["dropout"]

		self.gat_layer = GATConv(in_channels, out_channels, heads=num_heads, concat=concat,
							negative_slope=leaky_slope, dropout=dropout)

		# self.gcn_layer = GCNConv(in_channels, out_channels)

	def forward(self, x, edges):
		out_x = self.gat_layer(x, edges)
		return out_x

class ClassifierLayer(nn.Module):
	def __init__(self, type_, final_gat_out_dim, num_classes, dataset_name):
		super().__init__()
		self.type_ = type_
		self.dataset_name = dataset_name
		if type_ == "linear":
			self.drop = nn.Dropout(0.5)
			self.linear_layer1 = nn.Linear(final_gat_out_dim, final_gat_out_dim//2)
			self.linear_layer2 = nn.Linear(final_gat_out_dim//2, num_classes)
		else:
			self.class_reps = nn.Parameter(torch.randn(final_gat_out_dim, num_classes))

	def forward(self, x):
		if self.type_ == "linear":
			out1 = self.linear_layer1(x)
			return self.linear_layer2(self.drop(out1)), out1
		else:
			self.class_reps.data = F.normalize(self.class_reps.data, p=2, dim=1)
			return torch.mm(x, self.class_reps)


class Regularier:
	def __init__(self):
		self.sigmoid = torch.nn.Sigmoid()
		self.loss_func = torch.nn.L1Loss()

	def forward(self, node_embeds, Adj_block_idx):
		n1 = node_embeds[Adj_block_idx[0]]
		n2 = node_embeds[Adj_block_idx[1]]
		ones = torch.ones(Adj_block_idx.shape[1]).cuda()
		dot_prod = self.sigmoid( torch.sum(n1*n2, dim=1) )
		rec_loss = self.loss_func(dot_prod, ones)
		return rec_loss


class FewShotModel(nn.Module):
	def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, 
					final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device,
					num_gat_layers, gat_layer_params, knn_params, dataset, type_):
		super().__init__()

		self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, 
					final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device)
		self.gin_classifier = GINClassifier(num_layers, input_dim, hidden_dim, 
								output_dim, final_dropout, device)

		self.num_gat_layers = num_gat_layers
		self.gat_layer_params = gat_layer_params
		self.knn_params = knn_params

		self.gat_modules = torch.nn.ModuleList()
		for i in range(num_gat_layers):
			self.gat_modules.append(GATLayer(gat_layer_params[i], knn_params[i]))

		if gat_layer_params[num_gat_layers-1]["concat"] == 0:
			final_gat_out_dim = gat_layer_params[num_gat_layers-1]["out_channels"]
		else:
			final_gat_out_dim = gat_layer_params[num_gat_layers-1]["out_channels"] * \
								gat_layer_params[num_gat_layers-1]["heads"]

		self.method_type = type_
		if type_ == "train":
			num_classes = len(dataset.train_classes)
		elif type_ == "test":
			num_classes = len(dataset.test_classes)
		self.classifier = ClassifierLayer("linear", final_gat_out_dim, num_classes, dataset.dataset_name)

		self.regularier = Regularier()

	def forward(self, batch, dataset, args):
		if self.method_type == "train":
			pooled_h_layers, node_embeds, Adj_block_idx = self.gin(batch)
			x = pooled_h_layers[-1]
			gin_preds = self.gin_classifier(pooled_h_layers)
			
		else:
			with torch.no_grad():
				pooled_h_layers, node_embeds, Adj_block_idx = self.gin(batch)
				x = pooled_h_layers[-1]
				gin_preds = self.gin_classifier(pooled_h_layers)
		
		if self.method_type == "train":
			edges = dataset.create_train_knn_graph(x, batch, args)
		elif self.method_type == "test":
			edges = dataset.create_test_knn_graph(x, batch, args, gin_preds)
		
		x = torch.cat(pooled_h_layers[1:], dim=1)
		gat_outs = []
		x = F.normalize(x, p=2, dim=1)
		for i in range(self.num_gat_layers):
			x = self.gat_modules[i](x, edges)
			x = F.normalize(x, p=2, dim=1)
			gat_outs.append(x)

		return x, (node_embeds, Adj_block_idx), gin_preds, edges

	def train_(self, args, dataset):
		self.train()
		optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.01)
		criterion = nn.CrossEntropyLoss()
		super_class_loss_func = nn.CrossEntropyLoss()

		train_graphs = dataset.train_graphs

		for epoch in range(args.epochs+1):
			epoch_loss = []
			for iter_ in range(args.iters_per_epoch):
				selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
				batch_graph = [train_graphs[idx] for idx in selected_idx]
				batch_super_classes = torch.LongTensor([graph.super_class for graph in batch_graph]).cuda()
				output_embeds, (node_embeds, Adj_block_idx), gin_preds, edges = self.forward(batch_graph, 
										dataset, args)
				
				preds, _ = self.classifier(output_embeds)

				labels = torch.LongTensor([graph.label for graph in batch_graph]).cuda()

				# compute loss
				super_class_loss = super_class_loss_func(gin_preds, batch_super_classes)
				loss = criterion(preds, labels)
				optimizer.zero_grad()
				(loss + super_class_loss).backward()
				optimizer.step()
				epoch_loss.append(loss.detach().cpu().numpy())
				
			print("average loss for training epoch ", epoch, " = ", np.sum(epoch_loss)/len(epoch_loss))
			scheduler.step()

			if epoch % args.save_model_after == 0 and epoch > 0:
				print("Saving Main Model in training phase")
				torch.save(self.gin.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/main_model/gin_model/trained_{}.pth").format(epoch))
				torch.save(self.gin_classifier.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/main_model/gin_classifier/trained_{}.pth").format(epoch))
				torch.save(self.gat_modules.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/main_model/gat_modules/trained_{}.pth").format(epoch))


	def test_fine_tuning(self, args, dataset, device):
		self.gin.load_state_dict(torch.load("./checkpoints/{0}/main_model/gin_model/trained_{1}.pth".format(
					dataset.dataset_name, args.epochs)))
		self.gin_classifier.load_state_dict(torch.load("./checkpoints/{0}/main_model/gin_classifier/trained_{1}.pth".format(
					dataset.dataset_name, args.epochs)))
		self.gat_modules.load_state_dict(torch.load("./checkpoints/{0}/main_model/gat_modules/trained_{1}.pth".format(
				dataset.dataset_name, args.epochs)))
		print("GIN and GAT Modules loaded for fine tuning")

		test_fine_tuning_graphs = dataset.test_fine_tuning_graphs
		final_testing_graphs = dataset.final_testing_graphs
		
		self.train()
		
		self.gin.eval()
		self.gin_classifier.eval()

		# Fine tuning full model now
		optimizer = optim.Adam(self.parameters(), lr=args.fine_tune_lr, 
							weight_decay=args.fine_tune_weight_decay)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
		criterion = nn.CrossEntropyLoss()

		for epoch in range(args.fine_tune_epochs+1):
			epoch_loss = []
			for iter_ in range(args.fine_tune_iters_per_epoch):
				batch_graph = test_fine_tuning_graphs
				output_embeds, (node_embeds, Adj_block_idx), gin_preds, edges = self.forward(batch_graph, 
									dataset, args)
				
				preds, _ = self.classifier(output_embeds[:len(test_fine_tuning_graphs)])
				
				labels = torch.LongTensor([ graph.label for graph in batch_graph[:len(test_fine_tuning_graphs)] ]).cuda()

				#compute loss
				loss = criterion(preds, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				epoch_loss.append(loss.detach().cpu().numpy())

			print("Average loss for fine-tuning epoch ", epoch, " = ", np.sum(epoch_loss)/len(epoch_loss))
			scheduler.step()

			if epoch % args.fine_tune_save_model_after == 0 and epoch > 0:
				print("Saving Main Model for fine tuning")
				torch.save(self.gin.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/fine_tuned_test/gin_model/trained_{}.pth").format(epoch))
				torch.save(self.gin_classifier.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/fine_tuned_test/gin_classifier/trained_{}.pth").format(epoch))
				torch.save(self.gat_modules.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/fine_tuned_test/gat_modules/trained_{}.pth").format(epoch))
				torch.save(self.classifier.state_dict(),("./checkpoints/" + dataset.dataset_name + \
								"/fine_tuned_test/classifier/trained_{}.pth").format(epoch))


	def test(self, args, dataset, device):
		self.gin.load_state_dict(torch.load("./checkpoints/{0}/fine_tuned_test/gin_model/trained_{1}.pth".format(
					dataset.dataset_name, args.fine_tune_epochs)))
		self.gin_classifier.load_state_dict(torch.load("./checkpoints/{0}/fine_tuned_test/gin_classifier/trained_{1}.pth".format(
					dataset.dataset_name, args.fine_tune_epochs)))
		self.gat_modules.load_state_dict(torch.load("./checkpoints/{0}/fine_tuned_test/gat_modules/trained_{1}.pth".format(
				dataset.dataset_name, args.fine_tune_epochs)))
		self.classifier.load_state_dict(torch.load("./checkpoints/{0}/fine_tuned_test/classifier/trained_{1}.pth".format(
				dataset.dataset_name, args.fine_tune_epochs)))
		print("Loaded model for testing")

		self.eval()
		self.gin.eval()
		self.gin_classifier.eval()
		self.gat_modules.eval()
		
		final_testing_graphs = dataset.final_testing_graphs
		test_fine_tuning_graphs = dataset.test_fine_tuning_graphs

		with torch.no_grad():
			accuracy = []
			for run in range(args.num_testing_runs):
				run_accuracy = []
				random.shuffle(final_testing_graphs)
				
				for sample in final_testing_graphs[:args.num_inference_graphs]:
					batch_graph = test_fine_tuning_graphs + [sample]
					
					tr_len = len(test_fine_tuning_graphs)
					output_embeds, (node_embeds, Adj_block_idx), gin_preds, edges = self.forward(batch_graph, 
										dataset, args)
					
					preds, emb = self.classifier(output_embeds)
					y_preds = torch.argmax(preds, dim=1).cpu().numpy()
					labels = np.array([graph.label for graph in batch_graph]).astype(np.int32)
					acc = (y_preds[tr_len:] == labels[tr_len:]).sum() / (labels.shape[0] - tr_len)
					run_accuracy.append(acc)
				accuracy.append(np.mean(run_accuracy))
				
		print("Current run Testing accuracy = ", np.mean(accuracy))
		return np.mean(accuracy)