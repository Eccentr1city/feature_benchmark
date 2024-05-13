import json
from multiprocessing import Pool
import os
import re

model = 'gpt2-small'
autoencoder_layers = [2, 6]
autoencoder_bases = [
    'neurons',
    'res_scefr-ajt',
    'res_scl-ajt',
    'res-jb',
]

pos_classify_threshold = 0.05

def num_layers(basis):
    return 3072 if basis=='neurons' else 24576

def find_first_number(text):
    # Return the first number in a string
    match = re.search(r'\b\d+(\.\d+)?', text)
    if match:
        return float(match.group(0))
    else:
        # raise Exception("No valid model prediction")
        return None 

def parse_binary_response(text):
    # Search for the strings "high" and "low" and return whichever occurs first
    match = re.search(r'\b(high|low)\b', text, flags=re.IGNORECASE)
    if match:
        return 1 if match.group(0).lower() == 'high' else 0
    else:
        # raise Exception("No valid model prediction")
        return None 

def run_in_parallel(func, args_list):
    """
    Returns [func(arg1), func(arg2), ..., func(argn)] 
    where args_list is [arg1, arg2, ..., argn] and computes the function calls in parallel.
    """
    with Pool() as pool:
        results = pool.map(func, args_list)
    return results

def save_json_results(results, filename, indent=4):
    with open(filename, 'w') as f:
        if indent:
            json.dump(results, f, indent=indent)
        else:
            json.dump(results, f)

def load_json_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def print_json_tree(data, indent=''):
    if isinstance(data, dict):
        print("\n" + indent + '{', end='')
        for key in data:
            print("\n" + indent + str(key) + (' []' if isinstance(data[key], list) else ''), end='')
            print_json_tree(data[key], indent + '    ')
        print('\n' + indent + '}', sep='\n')
    elif isinstance(data, list):
        if data:
            print_json_tree(data[0], indent + '    ')
            for _ in range(3):
                print("\n" + indent + '.', end='')
 
def resave_organized_modeldata():
    model = 'gpt2-small'
    for layer in autoencoder_layers:
        for basis in autoencoder_bases:

            directory = f'{model}/{layer}' if basis == 'neurons' else f'{model}/{layer}-{basis}'
            new_directory = f'{model}-organized/{layer}' if basis == 'neurons' else f'{model}-organized/{layer}-{basis}'

            all_feature_data = {}
            files_in_directory = os.listdir(directory)
            for file in files_in_directory:
                print(f"{directory}/{file}")
                feature_data = load_json_results(f"{directory}/{file}")
                for elem in feature_data:
                    feature_id = int(elem['index'])
                    elem['activations'] = sorted(elem['activations'], key=lambda example: float(example['maxValue']), reverse=True)
                    all_feature_data[feature_id] = elem

            for feature_id, feature_data in all_feature_data.items():
                if not os.path.exists(new_directory):
                    os.makedirs(new_directory)
                save_json_results(feature_data, f"{new_directory}/{feature_id}.json", indent=0)

resave_organized_modeldata()

