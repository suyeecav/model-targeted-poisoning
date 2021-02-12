# About
This folder of maintains code relevant to DNN-models

# Install Dependencies
This code was tested on `python 3.6.12`
You can directly install the dependencies by running the following command:
```
pip install -r requirements.txt
```

# Run the Code
Please follow the instructions below to reproduce the results shown in the paper:

1. Generate stratified splits for MNIST-17
```
python split_mnist.py
```

2. Generate the target classifiers by running the command below.
```
python generate_target_dnn.py
```

4. To run our attack, run the following command:
```
python mtp_lookup.py --poison_model_path <POISON_MODELPATH> --verbose
```

%. To test how well this poison data works against different seeds, run
```
python generate_target_dnn.py --use_given_data --poison_path <POISON_DATA_PATH>
```

We experiment with the following seeds when running our attack:
<br>
`2021, 4, 16, 793, 80346`

We experiment with the following seeds when evaluating generated poison data against different weight initializations:
<br>
`24, 105, 418, 666, 3309, 42, 190, 762, 7, 3000`


# Analyzing trends

Useful statistics (loss values, model norms) will be logged in realtime in the `data/logs` folder, and can be browsed with Tensorboard. Generated poison data is also stored in this folder as poisondata.npz
