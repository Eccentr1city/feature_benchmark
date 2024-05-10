import json
from utils import *
import numpy as np
import pprint
import random
import requests
import time

def fetch_and_parse_json(url):
    print("WARNING: Querying Neuronpedia API")
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}... Maybe the internet is down or your feature id is not valid?")

def fetch_feature_data(args):
    feature_id, feature_data = args
    ## This really just has to return the parsed json with "activations" key and "explanations" key
    return feature_data[feature_id]
    # return fetch_and_parse_json(f"https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/{feature_id}")

def worker(feature, output):
    result = fetch_feature_data(feature)
    output.put(result)

def get_dict_from_example(example):
    elem = {
        'max_value': example['maxValue'],
        'max_value_token_index': example['maxValueTokenIndex'],
        'sentence_string': ''.join(example['tokens']),
        'max_token': example['tokens'][example['maxValueTokenIndex']],
        'tokens': example['tokens'],
        'values': example['values'],
    }
    return elem

## Could add different model or layer parameters to get more from neuronpedia
def get_pos_neg_examples(feature_id, feature_data, num_pos, num_neg, neg_type, randomize_pos_examples = True, seed=42):
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
    assert isinstance(feature_id, int) and feature_id >= 0, 'Invalid feature_id'
    assert isinstance(num_pos, int) and num_pos >= 0, 'Invalid num_pos'
    assert isinstance(num_neg, int) and num_neg >= 0, 'Invalid num_neg'
    assert neg_type in ['self', 'others'], 'Invalid neg_type'

    # Get feature parsed_json, description and highest activation
    parsed_json = fetch_feature_data((feature_id, feature_data))
    desc = parsed_json['explanations'][0]['description']
    highest_activation = parsed_json['activations'][0]['maxValue']

    # Asserts for positive examples
    assert len(parsed_json['activations']) >= num_pos, f"num_pos={num_pos} is greater than number of activations for feature {feature_id} on neuronpedia.org"
    assert parsed_json['activations'][num_pos]['maxValue'] >= 0.1*highest_activation, f"The num_pos = {num_pos}th example for feature {feature_id} on neuronpedia.org has a maxValue of {parsed_json['activations'][num_pos]['maxValue']} which is less than half the highest activation of {highest_activation} for feature {feature_id}"
    
    # Calculate pos
    pos = []

    for example in parsed_json['activations']:
        if example['maxValue'] >= 0.1*highest_activation:
            elem = get_dict_from_example(example)
            pos.append(elem)
    
    if randomize_pos_examples:
        random.seed(seed)
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
        np.random.seed(seed)
        neg_features = np.random.choice(len(feature_data) - 1, size=num_neg, replace=False)
        for i in range(len(neg_features)):
            if neg_features[i] >= feature_id:
                neg_features[i] += 1
            
            
        neg_data = [fetch_feature_data((neg_feature, feature_data)) for neg_feature in neg_features]

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

    random.seed(seed)
    random.shuffle(neg)
    assert len(neg) >= num_neg, f"num_neg={num_neg} is greater than the number of negative examples available for feature {feature_id} if neg_type=self"
    neg = neg[:num_neg]
            
    # Return the values
    return desc, pos, neg, highest_activation

if __name__ == "__main__":
    # pprint.pprint(get_pos_neg_examples(1, 3, 3, 'others'))
    # save_json_results(results, 'feat1.json')

    # results = load_json_results('feat1.json')
    # for result in results:
    #     print(result['index'])
    #     pprint.pprint(result)
    #     break

    # print(len(results))

    start = time.time()

    # for i in range(100):
    #     get_pos_neg_examples(i, feature_data, 3, 3, 'others')
    # # fetch_feature_data(2)
    # # fetch_feature_data(3)

    end = time.time()
    print(f"Time: {end - start}")

