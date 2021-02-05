# About
This repository maintains code for the paper "Poisoning Attacks with Provable Convergence to Convex Target Models". The KKT attack is adapted from its [original repository](https://github.com/kohpangwei/data-poisoning-journal-release).
# Install Dependencies
The program requires the following key dependencies:
`python 2.7`, `numpy`, `cvxpy (version 0.4.11)`, `scikit-learn`, `scipy`, `matplotlib`. You can directly install the dependencies by running the following command:
```
pip install -r requirements.txt
```

# Run the Code
Please follow the instructions below to reproduce the results shown in the paper:
1. unzip the file `files.zip` and you will see folder `files`, which contains the Adult and MNIST-17 datasets used for evaluation in the paper. In addition, we also provide the target classifiers for each dataset in the folder `files/target_classifiers`.
2. Skip this step if you wish to use the target classifiers we provide. Else, you can generate the target classifiers by running the command below. To generate target classifiers for the MNIST-17 dataset, replace `adult` with `mnist_17` in the command below. In the paper, we also improved the target model generation process for the MNIST-17 dataset and if you wish to use improved target model, add `--improved` in the command below.
```
python generate_target_theta.py --dataset adult
```

3. To run our attack, please use the command below. Again, replace `adult` with `mnist_17` to run the attack on MNIST-17 dataset. For the MNIST-17 dataset, if you wish to attack the improved target classifier, add `--improved` in the command below. By feeding different values to `--rand_seed`, we can repeat the attack process for multiple times and obtain more stable results. The Adult results in the paper are obtained by feeding `12`,`23`,`35`,`45` individually to `--rand_seed`. For MNIST-17 dataset, we feed `12`,`23`,`34`,`45` individually to `--rand_seed`.
```
python run_kkt_online_attack.py --rand_seed 12 --dataset adult
```

4. Once the attack is finished, run the following command to obtain the averaged results of the attack, which will be saved in directory `files/final_reslts` in `.csv` form. Replace dataset if necessary and if you used different random seeds for `--rand_seed` from above, please change the random_seeds specified in the source file.
```
python process_avg_results.py --dataset adult
```


5. To reproduce the figures in the paper, run the following command. Replace the dataset if necessary and also be careful if the random seeds are different from the ones used above and change accordingly in the source file. 
```
python plot_results.py --dataset adult
```

