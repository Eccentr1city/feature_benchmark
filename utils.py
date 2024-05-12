import json
from multiprocessing import Pool
import re

model = 'gpt2-small'
autoencoder_layers = [2, 6]
autoencoder_bases = [
    'neurons',
    'res_scefr-ajt',
    'res_scl-ajt',
    'jb',
]

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

def save_json_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

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
 