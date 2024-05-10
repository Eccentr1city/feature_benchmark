# feature_benchmark

There are three main steps to our framework:
1. Getting the positive and negative examples for a given feature.
2. Predicting the activations of the features by running experiments.
3. Analyzing the results of the experiments.

So, we have four files to consolidate the core functions we like and use:

1. getting_examples.py
   1. Here we have the function get_pos_neg_examples() which is responsible for getting the positive and negative examples for a given feature.
2. predict_activations.py
   1. Here we have the function predict_activations() which is responsible for predicting the activations of the features.
   2. We also have run_experiments() which is responsible for running the experiments for a given number of features.
3. analyze_results.py
   1. Here we (will) have the functions useful for analyzing the results of expeeriments.
4. utils.py
   1. Helper util functions that don't belong in a particular file

Finally, main.ipynb is for playing around with new things and doing science. I would have kept more functions here, but python has safeguards against using multiprocessing to call functions in the same file as the base file (otherwise you might infinitiely spawn processes), so I decided it was time to make them organized into their own files.

All results run from run_experiments are saved in the results folder. They can be easily loaded with JSON.