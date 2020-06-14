# FEW-SHOT LEARNING ON GRAPHS VIA SUPERCLASSES BASED ON GRAPH SPECTRAL MEASURES

Source code for our [ICLR 2020](https://iclr.cc/Conferences/2020) paper: [FEW-SHOT LEARNING ON GRAPHS VIA SUPER-CLASSES BASED ON GRAPH SPECTRAL MEASURES](https://openreview.net/forum?id=Bkeeca4Kvr). Associated Blogpost can be found [here](https://medium.com/@cs17btech11019/few-shot-learning-on-graphs-f6312a9e9de5).

### Requirements
Please create a virtual environment for smoother functioning and to avoid any dependency issues. Please refer to [Managing Virtual Environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for details on creating virtual environment.
After activating the virtual environment, run the following command to install all the required libraries and modules(Note: the requirements.txt file contains some additional libraries which you may or may not need) - 

     `pip install -r requirements.txt`

## Directory Structure 

The `datasets` directory contains the graphs for the datasets - Letter-High and TRIANGLES used for this paper. Inside each dataset sub-directory, there is a file named `class_prototype_numbers.json` which contains the precomputed class prototypes as given in the paper. The file `make_class_prototypes.py` contains the code to create the class prototype graphs. The `train_test_classes.json` contains the training-testing class splits used. The sub-directory `json_format` contains the graphs in json format.

The `src` directory contains the source code for this paper. The `main.py` file contains the initiation code, starting with the argument parser. The important arguments are -

'--dataset_name' : Name of the dataset

'--num_layers' : Number of layers in GIN model

'--hidden_dim' : Number of dimensions in GIN's MLP layers

'--graph_pooling_type' : Pooling over nodes to get graph embedding: sum or average

'--neighbor_pooling_type' : Pooling over neighboring nodes: sum, average or max

'--num_gat_layers' : Number of GAT layers

'--gat_heads' : Number of heads per GAT layer

'--batch_size' : size of mini-batch

'--n_shot' : The shot scenario to evaluate upon

'--knn_value' : The number of neighbors per graph representation node in super-graph

'--train_clusters' : The number of super-classes

Rest all the arguments are self-explanatory.

## Evaluation

To evaluate the model for 20-shot on TRIANGLES datasets run - 

`python3 main.py --dataset_name TRIANGLES --batch_size 64 --knn_value 2`

An example usage for the above has been provided in the jupyter notebook: [src/example.ipynb](https://github.com/chauhanjatin10/GraphsFewShot/blob/master/src/example.ipynb) .

The `src/checkpoints` directory stores the trained weights for the model. Run the bash file `clear_checkpoints.sh` to clear existing checkpoints. The `dataloader.py` file contains the class for loading data and creating splits for fine-tuning as well as testing. The base code for the files `graphcnn.py`, `mlp.py` and `util.py` have been taken from the original implementation of the [GIN paper](https://github.com/weihua916/powerful-gnns) and further modified for our purpose. 

## Citation

Please cite the following paper if you use this code in your work.
     
     @inproceedings{chauhan2020fewshot,
     title=FEW-SHOT LEARNING ON GRAPHS VIA SUPER-CLASSES BASED ON GRAPH SPECTRAL MEASURES,
     author={Jatin Chauhan and Deepak Nathani and Manohar Kaul},
     booktitle={International Conference on Learning Representations},
     year={2020},
     url={https://openreview.net/forum?id=Bkeeca4Kvr}
     }
