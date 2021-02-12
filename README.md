# About
This repository maintains code for the model-targeted poisoning attacks. The KKT attack is adapted from its [original github repository](https://github.com/kohpangwei/data-poisoning-journal-release). Our experiments on deep neural networks are in a separate folder `dnn` and you can find more instructions inside the folder.
# Install Dependencies
The program requires the following key dependencies:
`python 2.7`, `numpy`, `cvxpy (version 0.4.11)`, `scikit-learn`, `scipy`, `matplotlib`. You can directly install the dependencies by running the following command:
```
pip install -r requirements.txt
```

# Run the Code
Please follow the instructions below to reproduce the results shown in the paper:
1. unzip the file `files.zip` and you will see folder `files`, which contains the Adult, MNIST-17 and Dogfish datasets used for evaluation in the paper. In addition, we also provide the target classifiers for each dataset in the folder `files/target_classifiers`.
2. Skip this step if you wish to use the target classifiers we provide. Else, you can generate the target classifiers by running the command below. To generate target classifiers for other datasets, replace `adult` with `mnist_17` or `dogfish` in the command below. To obtain results on logistic regression model, replace `svm` with `lr`. In the paper, we also improved the target model generation process for the MNIST-17 dataset and the SVM model, and if you wish to use improved target model, add `--improved` in the command below.
```
python generate_target_theta.py --dataset adult --model_type svm
```

3. To run our attack, please use the command below. Again, replace `adult` with `mnist_17` or `dogfish` to run the attack on other datasets. Replace `svm` with `lr` to run the attack on logistic regression model. For the MNIST-17 dataset, if you wish to attack the improved target classifier, add `--improved` in the command below. By feeding different values to `--rand_seed`, we can repeat the attack process for multiple times and obtain more stable results. Results in the paper can be reproduced by feeding the seeds `12`,`23`,`34`,`45` individually to `--rand_seed`.
```
python run_kkt_online_attack.py --rand_seed 12 --dataset adult --model_type svm
```

4. Once the attack is finished, run the following command to obtain the averaged results of the attack, which will be saved in directory `files/final_reslts` in `.csv` form. Replace dataset if necessary and if you used different random seeds for `--rand_seed` from above, please change the random_seeds specified in the source file. You can find the number of poisoning points used and also the computed lower bound in the `csv` file. 
```
python process_avg_results.py --dataset adult --model_type svm
```

5. To generate the test accuracies (after poisoning) reported in Table 1 and Table 2 in the paper, run the following command to get the averaged results. Change datasets and model types if necessary.
```
python generate_table.py --dataset adult --model_type svm 
```  

6. To reproduce the figures in the paper, run the following command. Replace the dataset if necessary and also be careful if the random seeds are different from the ones used above and change accordingly in the source file. 
```
python plot_results.py --dataset adult --model_type svm
```

