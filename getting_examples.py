import json
from multiprocessing import Pool
import numpy as np
import pprint
import random
import requests
import time


def fetch_and_parse_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}... Maybe the internet is down or your feature id is not valid?")

def fetch_feature_data(feature_id):
        return fetch_and_parse_json(f"https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/{feature_id}")

# def get_activation_data_for_feature(url):
#     """
#     STRUCTURE OF cleaned_data

#     exapmles: list of examples where each example is {'maxValue': int, 'maxValueTokenIndex': int, 'tokens': list, 'values': list}
#         tokens and values have same length where values has the activation on each token

#     explanations: contains things like
#         'author': {'country': 'US', 'name': 'gpt-3.5-turbo'},
#         'description': 'phrases related to Q&A sessions or dialogues involving opinions and discussions',
#         'layer': '9-res-jb',
#         'modelId': 'gpt2-small', 
#     """
#     parsed_json = fetch_and_parse_json(url)

#     cleaned_data = {}
#     cleaned_data['explanations'] = parsed_json['explanations']
#     examples = []
#     for example in parsed_json['activations']:
#         examples.append(
#             {
#                 'max_value': example['maxValue'],
#                 'max_value_token_index': example['maxValueTokenIndex'],
#                 'tokens': example['tokens'],
#                 'values': example['values'],
#             }
#         )
#         # assert len(example['tokens']) == len(example['values']), "Error"

#     cleaned_data['examples'] = examples

#     return cleaned_data

def worker(feature, output):
    result = fetch_feature_data(feature)
    output.put(result)

def get_pos_neg_examples(feature_id, num_pos, num_neg, neg_type, randomize_pos_examples = False):
    """
    Input:
    feature_id: int >= 0
    num_pos: int >= 0
    num_neg: int >= 0
    neg_type: "self" or "others"

    Output dictionary:
    pos_data: list
    neg_data: list
    """

    # Input assertions
    assert isinstance(feature_id, int) and feature_id >= 0, 'Invalid feature_id'
    assert isinstance(num_pos, int) and num_pos >= 0, 'Invalid num_pos'
    assert isinstance(num_neg, int) and num_neg >= 0, 'Invalid num_neg'
    assert neg_type in ['self', 'others'], 'Invalid neg_type'

    # Get feature parsed_json, description and highest activation
    parsed_json = fetch_feature_data(feature_id)
    desc = parsed_json['explanations'][0]['description']
    highest_activation = parsed_json['activations'][0]['maxValue']


    # Asserts for positive examples
    assert len(parsed_json['activations']) >= num_pos, f"num_pos={num_pos} is greater than number of activations for feature {feature_id} on neuronpedia.org"
    assert parsed_json['activations'][num_pos]['maxValue'] >= 0.1*highest_activation, f"The num_pos = {num_pos}th example for feature {feature_id} on neuronpedia.org has a maxValue of {parsed_json['activations'][num_pos]['maxValue']} which is less than half the highest activation of {highest_activation} for feature {feature_id}"
    
    # Calculate pos
    pos = []

    ####### IMPLEMENT THIS SOOON ############
    # pos_indices = np.random.choice(range(parsed_json['activations'][num_pos]['maxValue']), num_pos) if not randomize_pos_examples else np.random.choice(range(num_pos), num_pos)
    pos_indices = list(range(num_pos))
    random.shuffle(pos_indices)

    for i in pos_indices:
        example = parsed_json['activations'][i]
        elem = {
            'max_value': example['maxValue'],
            'max_value_token_index': example['maxValueTokenIndex'],
            'sentence_string': ''.join(example['tokens']),
            'max_token': example['tokens'][example['maxValueTokenIndex']],
            'tokens': example['tokens'],
            'values': example['values'],
        }
        pos.append(elem)
    

    # Calculate neg
    neg = []

    neg_features = np.random.choice(10000, size=3*int(num_neg**0.5), replace=False)
    if feature_id in neg_features:
        neg_features.remove(feature_id)

    with Pool() as pool:
        neg_data = pool.map(fetch_feature_data, neg_features)

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
    neg = neg[:num_neg]
            
    # Return the values
    return desc, pos, neg, highest_activation

if __name__ == "__main__":
    start = time.time()
    pprint.pprint(get_pos_neg_examples(1, 3, 3, 'self'))
    end = time.time()
    print(f"Time: {end - start}")

