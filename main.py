from getting_examples import *
from predict_activations import *
from utils import *
import json
import pprint


if __name__ == '__main__':

    with open(f'feats.json', 'r') as file:
        feature_data = json.load(file)

    result_1 = run_experiments(
        num_features=100, 
        feature_data=feature_data,
        test_pos=5, # Experiment with
        test_neg=5, # Experiment with
        show_pos=0, # Experiment with
        show_neg=0, # Experiment with
        neg_type='self', # Experiment with
        binary_class=True, # Experiment with
        show_max_token=False, # Experiment with
        num_completions=1, # Experiment with
        debug=False, 
        randomize_pos=True, 
        seed=42,
        save_location='test'
    )

    # result_2 = run_experiments(
    #     num_features=100, 
    #     feature_data=feature_data,
    #     test_pos=5, # Experiment with
    #     test_neg=5, # Experiment with
    #     show_pos=0, # Experiment with
    #     show_neg=0, # Experiment with
    #     neg_type='others', # Experiment with
    #     binary_class=False, # Experiment with
    #     show_max_token=False, # Experiment with
    #     num_completions=1, # Experiment with
    #     debug=False, 
    #     randomize_pos=True, 
    #     seed=42,
    #     save_location='test'
    # )

