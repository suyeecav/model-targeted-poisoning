# subpop_attack

<!-- To generate the synthetic data for visualization purpose, go to `synthetic_data` folder and run the `generate_and_visualize.py` file. By changing the value of `class_sep` in the python script, you can control the distance of each cluster and the distance of two classes. 

The folder `subpopulation_poison` contains files of adult dataset and the label flipping attack, and the attacked models are logistic regression and MLP. To run these files, execute `subpopulation_adult_compactness.py` file, and change the poison ratio if necessary. 

The folder `data-poisoning-journal-release` contains files for strong KKT-attack, adapted from the [paper](https://arxiv.org/pdf/1811.00741.pdf). To run the subpopulation attack, execute `python run_kkt_attack_sub.py kkt-standard --dataset mnist_17`, and change the dataset name if needed. If you want to run the label flipping attack, add an additional aruguement `--label_flip_baseline` to the command line. Subpopulations for MNIST_17 and Adult are clusters. For Enron dataset, subpopulation is defined based on features. To run the subpopulation attack on Enron, run `python run_kkt_attack_sub.py kkt-standard`. Label flippping attack can be run by adding additional term `--label_flip_baseline`. If you want to run the attack on whole distribution, run `python run_kkt_attack.py kkt-standard --dataset mnist_17`. 

Note for strong KKT-attack: it requires some special convex optimization solvers like cvxpy, there you can create a virtual python environment (e.g., conda create XXX) and install the required libraries. Details about the packages can be found [here](https://github.com/kohpangwei/data-poisoning-journal-release). -->

## Experiment Setup
Our adaptive online attack and the KKT attack require a convex optimization tool named `cvxpy` and a solver named `Gurobi`. We recommend to create a virtual python environment (e.g., conda create XXX) and install the required libraries. Details about the packages can be found [here](https://github.com/kohpangwei/data-poisoning-journal-release). 

This repository contains the code for reproducing results of adaptive poisoning attacks. First, please download the related data files from [here]() and place the extracted folder `files` in the same directory as the source files. 

## Generate Valid Target Classifiers and Compare 
To generate the target classifier using the optimization scheme presented in the paper, please run the using the command and replace the dataset name if needed (i.e., replace the `adult` with `mnist_17`)
```
python generate_target_theta_optimization.py --dataset adult
``` 
When all the valid target classifiers (satisfy attacker objective) are generated, you can also check their individual performances to validate whether thetas with lower loss on clean training set indeed require lower number of points. To do so, run the following command and repace the dataset name if needed.
```
python check_target_thetas.py --dataset adult
```

## Run the Actual Attacks and Obtain the Lower Bound
To run both the KKT attack and the adaptive online attack proposed in this paper, run the following command and replace the dataset name if needed.
```
python run_kkt_online_attack.py --dataset adult
``` 
By default, the attack will be performed to attack both the actual target model (obtained from the optimization scheme in the paper), classifiers produced by the KKT and adaptive online attacks in the paper. If you hope to only attack the actual target classifier, set the keyword `target_model` as `real`, and set `target_model` as `kkt` to attack the kkt attack produced model, and `ol` to attack classifier produced by our adaptive online attack.

