# NEON - Multi Neural Network Compression Agent

This repository is the official implementation of NEON from "Multi-objective pruning of dense neural networks using deep reinforcement learning
". [https://doi.org/10.1016/j.neunet.2022.06.018](https://www.sciencedirect.com/science/article/abs/pii/S0020025522008222)

## Preparing the workstation
This project is using `conda` as the package manager. 
In order to create a new environment with the required packages you should run the following command:
```
conda create --name <env> --file requirements.txt
``` 

## Training base networks
You should use notebook file `Train Base Networks.ipynb` and set up the following steps:
1. Change `PATH` variable to the location of the datasets files. The main dataset folder should include a folder for 
each dataset (with the name of the dataset name) and `csv` file contains the dataset.

2. Change the list `dataset_names` to include all the datasets names you wish to train.

3. Run the notebook. This notebook will create and train 30 networks with random architecture for each dataset. The output of this script will create: 1) 30 trained networks (`.pt` files). 2) The dataset after the preprocessing - `X_train.csv`, `Y_train.csv` files.
  

## Running the agent
### Prepare folder structure
In order to train NEON agent on the a specific dataset and its network, you should have a folder with the dataset and networks.
You should have a folder with a named `OneDatasetLearning`, with a subfolder named `Classification`. Inside `Classification` folder you should include a folder for each dataset (with the dataset name). Each dataset folder should include all the `.pt` (trained networks) files and the dataset's `X_train.csv`, `Y_train.csv` `.csv` file.

Moreover, you should create another folders structure that will contain the output files from the training. You should create in the root folder a new folder named `models` and a sub-folder named `Reinforce_One_Dataset`. 
 
### Train NEON agent
You should use `a2c_agent_reinforce_runner.py` file to train NEON agent. when running this file, you should provide the following hyper-parameters:
* `--dataset_name` - The name of the dataset to train - `string`
* `--learn_new_layers_only` - Whether to train only the  new layer after the compression or the whole network. If you choose to train the entire network, the training time is much longer. We recommend to use this hyper-parameter with `True` value.
* `--split` - Whether to split the networks to train and test sets. In the first time you train an agent on a new dataset folder you must use this hyper-parameter with `True` value. If you train another agent on the same dataset you can set this parameter to `False`. This process creates `train_models.csv` and `test_models.csv` files.
* `--allowed_reduction_acc` - This hyper-parameters defines the “permissible” reduction in performance. We recommend to use this value with 1 or 5.
* `--increase_loops_from_1_to_4` - This hyper-parameter controls whether the agent can compress each layer 4 times instead of 1.
* `--seed` (int) - This hyper-parameter defines the seed to be used by pytorch and numpy libraries. Default value is 0.


The output of this training will produce 2 files in `./models/Reinforce_One_Dataset/` with the following structure:
`results_Agent_[DatasetName]_new_layers_only_[hp value]_acc_reduction_[hp value][_with_loop]_[train/test].csv`. (hp = hyper-parameter).
* `DatasetName` - The chosen dataset name
* `_new_layers_only_[hp value]` - The chosen hyper-parameter value whether training the only the new layer.
* `acc_reduction_[hp value]` - The chosen allowed reduction of the accuracy.
* `[_with_loop_]` - Will be included in the file name if `--increase_loops_from_1_to_4` is `True`.
* `[train/test]` - Each training produces 2 files: one for the train networks and one for the test networks.

Each result file includes the new architecture of each network, the original and compressed networks' accuracies and the original and compressed parameters.

## Paper training Agent hyper-parameters:
### Agent with 4 iterations over each architecture:
```
a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --increase_loops_from_1_to_4 --allowed_reduction_acc=1

a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --increase_loops_from_1_to_4 --allowed_reduction_acc=5

a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --increase_loops_from_1_to_4 --allowed_reduction_acc=50
``` 
      
### Agent with 1 iteration over each architecture:
```
a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --allowed_reduction_acc=1

a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --allowed_reduction_acc=5

a2c_agent_reinforce_runner.py --dataset_name=[Insert here the dataset name] --learn_new_layers_only=True --allowed_reduction_acc=50
``` 

## Citation
If you found this work useful, please cite the following related article:
```
title = {Multi-objective pruning of dense neural networks using deep reinforcement learning},
journal = {Information Sciences},
volume = {610},
pages = {381-400},
year = {2022},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2022.07.134},
url = {https://www.sciencedirect.com/science/article/pii/S0020025522008222},
author = {Lior Hirsch and Gilad Katz},
keywords = {Pruning, Deep reinforcement learning},
abstract = {Network pruning aims to reduce the inference cost of large models and enable neural architectures to run on end devices such as mobile phones. We present NEON, a novel iterative pruning approach using deep reinforcement learning (DRL). While most reinforcement learning-based pruning solutions only analyze the one network they aim to prune, we train a DRL agent on a large set of randomly-generated architectures. Therefore, our proposed solution is more generic and less prone to overfitting. To avoid the long-running times often required to train DRL models for each new dataset, we train NEON offline on multiple datasets and then apply it to additional datasets without additional training. This setup makes NEON more efficient than other DRL-based pruning methods. Additionally, we propose a novel reward function that enables users to clearly define their pruning/performance trade-off preferences. Our evaluation, conducted on a set of 28 diverse datasets, shows that the proposed method significantly outperforms recent top-performing solutions in the pruning of fully-connected networks. Specifically, our top configuration reduces the average size of the pruned architecture by ×24.59, compared to ×13.26 by the leading baseline, while actually improving accuracy by 0.5%.}
```

