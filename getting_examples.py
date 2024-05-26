import json
from utils import *
import numpy as np
import pprint
import random
import requests
import time

model_name = 'gpt2-small'
autoencoder_layers = [2, 6]
autoencoder_bases = [
    'neurons',
    'res_scefr-ajt',
    'res_scl-ajt',
    'res-jb',]


def fetch_and_parse_json(url):
    print("WARNING: Querying Neuronpedia API")
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}... Maybe the internet is down or your feature id is not valid?")


def fetch_feature_data(layer, basis, feature_id):
    assert layer in autoencoder_layers, f"Invalid layer: {layer} not in {autoencoder_layers}"
    assert basis in autoencoder_bases, f"Invalid model: {basis} not in {autoencoder_bases}"

    data_dir = f"{model_name}-organized/{layer}" if basis == 'neurons' else f"{model_name}-organized/{layer}-{basis}"
    
    with open(f"{data_dir}/{feature_id}.json", 'r') as f: #This is the step that takes a while (66 / 75 ms)
        feature_data = json.load(f)

    # assert result is not None, f"Feature id = {feature_id} does not exist in the data dump file {data_dir}/{filename}-{filename + skip}.json"
    assert int(feature_data['index']) == feature_id, f"Feature id = {feature_id} does not match the feature id shown in the data dump: {feature_data['index']}"
    return feature_data

    ## This really just has to return the parsed json with "activations" key and "explanations" key
    # return feature_data[feature_id]
    # return fetch_and_parse_json(f"https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/{feature_id}")


def features_exist(layer, basis, feature_id, depth = 5):
    data = fetch_feature_data(layer, basis, feature_id)
    if not data['explanations']:
        return False
    if data['activations'][depth]['maxValue'] >= data['activations'][0]['maxValue'] * pos_classify_threshold:
        return True
    return False


def worker(feature, output):
    result = fetch_feature_data(feature)
    output.put(result)


def get_dict_from_example(example):
    elem = {
        'sentence_string': ''.join(example['tokens']),
        'tokens': example['tokens'],
    }
    if 'recomputedValues' in example:
        elem['values'] = example['recomputedValues'][1:]
    else:
        elem['values'] = example['values']
    if 'maxValueTokenIndex' in example:
        elem['max_value_token_index'] = example['maxValueTokenIndex']
        elem['max_token'] = example['tokens'][example['maxValueTokenIndex']],
    elem['max_value'] = max(elem['values'])
    return elem


## Could add different model or layer parameters to get more from neuronpedia
def get_pos_neg_examples(feature_id, layer, basis, num_pos, num_neg, neg_type, randomize_pos_examples = True):
    """
    Input:
    feature_id: int >= 0
    num_pos: int >= 0
    num_neg: int >= 0
    neg_type: "self" OR "others"
    randomize_pos_examples: bool

    Output dictionary:
    desc: str
    pos_data: list of dicts
    neg_data: list of dicts
    highest_activation: float
    """

    # Input assertions
    assert isinstance(feature_id, int) and feature_id >= 0, f'Invalid feature_id {feature_id}'
    assert isinstance(num_pos, int) and num_pos >= 0, 'Invalid num_pos'
    assert isinstance(num_neg, int) and num_neg >= 0, 'Invalid num_neg'
    assert neg_type in ['self', 'others'], 'Invalid neg_type'

    # Get feature parsed_json, description and highest activation
    parsed_json = fetch_feature_data(layer, basis, feature_id)
    desc = parsed_json['explanations'][-1]['description']
    highest_activation = parsed_json['maxActApprox']

    # Asserts for positive examples
    assert len(parsed_json['activations']) >= num_pos, f"num_pos={num_pos} is greater than number of activations for feature {feature_id} in layer {layer} with basis {basis} on neuronpedia.org"
    assert parsed_json['activations'][num_pos]['maxValue'] >= highest_activation * pos_classify_threshold, f"The num_pos = {num_pos}th example for feature {feature_id} in layer {layer} with basis {basis} on neuronpedia.org has a maxValue of {parsed_json['activations'][num_pos]['maxValue']} which is less than {pos_classify_threshold} of the highest activation of {highest_activation} for feature {feature_id}"

    # Calculate pos
    pos = []

    for example in parsed_json['activations']:
        if example['maxValue'] >= 0.1*highest_activation:
            elem = get_dict_from_example(example)
            pos.append(elem)
    
    if randomize_pos_examples:
        random.shuffle(pos)

    pos = pos[:num_pos]
    

    # Calculate neg
    neg = []

    if neg_type == 'self':
        for example in parsed_json['activations']:
            if example['maxValue'] < 0.001*highest_activation:
                elem = get_dict_from_example(example)
                neg.append(elem)
    elif neg_type == 'others':
        neg_features = []
        while len(neg_features) < num_neg:
            f_id = int(np.random.choice(num_layers(basis), size=1, replace=False))
            if f_id != feature_id and f_id not in neg_features and features_exist(layer, basis, f_id):
                neg_features.append(f_id)  
            
        neg_data = [fetch_feature_data(layer, basis, neg_feature) for neg_feature in neg_features]

        # neg_data = run_in_parallel(fetch_feature_data, [(neg_feature_id, feature_data) for neg_feature_id in neg_features])

        for feature_parsed_json in neg_data:
            h_a = feature_parsed_json['activations'][0]['maxValue']
            for i in range(len(feature_parsed_json['activations'])):
                example = feature_parsed_json['activations'][i]
                if example['maxValue'] >= 0.1*h_a:
                    elem = {
                        'max_value': 0,
                        'sentence_string': ''.join(example['tokens']),
                        'tokens': example['tokens'],
                        'values': [0]*len(example['tokens']),
                    }
                    neg.append(elem)
                else:
                    break

    random.shuffle(neg)
    assert len(neg) >= num_neg, f"num_neg={num_neg} is greater than the number of negative examples available for feature {feature_id} if neg_type=self"
    neg = neg[:num_neg]
            
    # Return the values
    return desc, pos, neg, highest_activation

# If we have precomputed positive and negative examples in a file, use this one
def get_pos_neg_from_file(file_path, num_pos, num_neg, neg_type, randomize_pos_examples = True):
    assert neg_type in ['self', 'others'], 'Invalid neg_type'
    with open(file_path, 'r') as file:
        data = json.load(file)

    highest_activation = data['maxActApprox']
    desc = data['explanations'][-1]['description']
    if len(data['posActivations']) < num_pos:
        print('WARNING: Insufficient positive activations')
    if randomize_pos_examples:
        pos_examples = random.sample(data['posActivations'], max(num_pos, len(data['posActivations'])))
    else:
        pos_examples = data['posActivations'][:num_pos]
    pos = [get_dict_from_example(p) for p in pos_examples]
    neg_key = 'negSelfActivations' if neg_type == 'self' else 'negOtherActivations'
    neg_examples = random.sample(data[neg_key], max(num_pos, len(data[neg_key])))
    neg = [get_dict_from_example(p) for p in neg_examples]
    
    return desc, pos, neg, highest_activation


def recompute_activations(file_path, model, sae):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    strings = []
    activations = data['activations']
    index = int(data['index'])
    
    for activation in activations:
        strings.append(''.join(activation['tokens']))
    
    _, inner, _ = get_sae_activations(model, sae, strings)
    
    # Add recomputed values for current feature back to the JSON structure
    for i, activation in enumerate(activations):
        activation['recomputedValues'] = [x[index] for x in inner[i]]
    
    with open(file_path, 'w') as file:
        json.dump(data, file)


def sort_activations(file_path, pos_classify_threshold):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    posActivations = []
    negSelfActivations = []
    maxActApprox = data['maxActApprox']
    threshold = pos_classify_threshold * maxActApprox

    # Sort activations into posActivations and negSelfActivations
    for activation in data['activations']:
        if sum(activation['recomputedValues']) > threshold:
            posActivations.append(activation)
        else:
            negSelfActivations.append(activation)
    
    # Compute the averagePosActivation
    if posActivations:
        # Compute the elementwise average of recomputed positive activations (for mean-ablating)
        max_length = max(len(act['recomputedValues']) for act in posActivations)
        avg_matrix = np.array([np.pad(act['recomputedValues'], (0, max_length - len(act['recomputedValues'])), 'constant') for act in posActivations])
        averagePosActivation = np.mean(avg_matrix, axis=0).tolist()
    else:
        averagePosActivation = []

    # Add updated data back to the JSON file
    data['posActivations'] = posActivations
    data['negSelfActivations'] = negSelfActivations
    data['averagePosActivation'] = averagePosActivation
    with open(file_path, 'w') as file:
        json.dump(data, file)
    
    return len(posActivations), len(negSelfActivations)


# Precompute a set of negative examples stolen from other features and add them to the JSON files in the subset folder
def other_negative_activations(file_path, other_directory, model, sae, num_neg):
    with open(file_path, 'r') as file:
        data = json.load(file)
    feature_id = int(data['index'])
    max_act = data['maxActApprox']
    negOtherActivations = []

    files = [f for f in os.listdir(other_directory) if os.path.isfile(os.path.join(other_directory, f))]
    files.remove(os.path.basename(file_path))
    neg_features = random.sample(files, min(num_neg, len(files)))
    for nf in neg_features:
        files.remove(nf)
    
    for i, neg_feature in enumerate(neg_features):
        negative = False
        while not negative:
            with open(f"{other_directory}/{neg_feature}", 'r') as f:
                neg_data = json.load(f)
            neg_act = random.sample(neg_data['activations'][:10], 1)[0]
            neg_string = ''.join(neg_act['tokens'])
            _, inner, _ = get_sae_activations(model, sae, [neg_string])
            recomputed = [x[feature_id] for x in inner[0]]
            if sum(recomputed) < 0.01 * max_act:
                negative = True
            else:
                print('WARNING: Resampling negative activation')
                neg_feature = random.sample(files, 1)[0]
                files.remove(neg_feature)

            neg_act['recomputedValues'] = recomputed
            del neg_act['values']
            del neg_act['maxValue']
            del neg_act['minValue']
            del neg_act['maxValueTokenIndex']
            del neg_act['lossValues']
            negOtherActivations.append(neg_act)
    
    # Add updated data back to the JSON file
    data['negOtherActivations'] = negOtherActivations
    with open(file_path, 'w') as file:
        json.dump(data, file)
                
    return


def recompute_directory_activations(directory, other_directory, model, sae, recompute=True, re_sort=True, num_neg_others=10):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            if recompute:
                recompute_activations(file_path, model, sae)
                print(f'{filename} updated with recomputed activations')
            if re_sort:
                pos, self_neg = sort_activations(file_path, pos_classify_threshold)
                print(f'{filename} sorted into {pos} positive and {self_neg} self_negative activations')
            if num_neg_others:
                other_negative_activations(file_path, other_directory, model, sae, num_neg_others)
                print(f'{filename} gathered {num_neg_others} other_negative activations')



def sanity_checking_data_pipeline():
    """
    A function to sanity check that the data pipeline seems right and you're getting what you should from get_pos_neg_examples.
    
    Just spam run this and confirm the outputs look right. Vary the autoencoder basis and layer as well. Can hold seed constant for deterministic outputs.
    """
    layer= random.choice(autoencoder_layers)
    basis = random.choice(autoencoder_bases)
    neg_type='others'
    randomize_pos=True 
    num_pos=4
    num_neg=4
    seed=np.random.randint(0, 10000)
    num_features = 2

    feature_indices = []
    while len(feature_indices) < num_features:
        feature_index = int(np.random.choice(1000, 1, replace=False)[0])
        if feature_index not in feature_indices and features_exist(layer, basis, feature_index):
            feature_indices.append(feature_index)

    print(f"Layer: {layer}, Basis: {basis}")
    for feature_index in feature_indices:
        desc, pos, neg, high = get_pos_neg_examples(feature_index, layer, basis, num_pos=num_pos, num_neg=num_neg, neg_type=neg_type, randomize_pos_examples=randomize_pos, seed=seed)

        print('pos ' + str(feature_index))
        for elem in pos:
            print(elem['max_value'])
        print('neg ' + str(feature_index))
        for elem in neg:
            print(elem['max_value'])