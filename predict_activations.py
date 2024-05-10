import concurrent.futures
from functools import partial
from getting_examples import get_pos_neg_examples
import json
from multiprocessing import Pool
import numpy as np
from openai import OpenAI
import pprint
import random
import time
from utils import *

client = OpenAI()

def predict_activations(feature_index, test_pos=20, test_neg=20, show_pos=0, show_neg=0, binary_class=True, neg_type='others', show_max_token=False, num_completions=1, debug=False, randomize_pos=True, seed=42):
    # Get positive and negative examples of the feature activation
    num_pos = test_pos + show_pos
    num_neg = test_neg + show_neg
    description, pos_examples, neg_examples, highest_activation = get_pos_neg_examples(feature_index, num_pos=num_pos, num_neg=num_neg, neg_type=neg_type, randomize_pos_examples=randomize_pos, seed=seed)

    if binary_class:
        for sentence in pos_examples:
            sentence['max_value'] = 'high'
        for sentence in neg_examples:
            sentence['max_value'] = 'low'

    test_sentences = pos_examples[:test_pos] + neg_examples[:test_neg]
    show_sentences = pos_examples[test_pos:] + neg_examples[test_neg:]
    extra_data = {
        'description': description,
        'test_sentences': test_sentences,
        'show_sentences': show_sentences,
        'highest_activation': highest_activation
    }

    # Create a system prompt dependning on how many example sentences are provided
    system_prompt = f'You are evaluating an english description of an autoencoder feature. The description should correspond to sentences which result in high activation. The english description of the feature is: "{description}"\n'

    if show_pos or show_neg:
        system_prompt += f'Here are {show_pos + show_neg} examples of sentences and their corresponding activations:\n'
        for sentence in show_sentences:
            sentence_string = sentence['sentence_string']
            activation = sentence['max_value']
            if not binary_class:
                activation = round(sentence['max_value'], 2)
            system_prompt += f'Example: "{sentence_string}", Activation: {activation}'
            if show_max_token:
                max_token = sentence['max_token']
                max_token_index = sentence['max_value_token_index']
                system_prompt += f', Token with highest activation: "{max_token}" at token {max_token_index}'
            system_prompt += '\n'

    if binary_class:
        system_prompt += f'\nYou must predict the activation on a new sentence based off of the provided description. If the description matches the provided sentence, the activation will be high, while if it does not match the activation will low. \nYou MUST respond with either "high" or "low" and NO OTHER content.'
    else:
        system_prompt += f'\nThe value of the highest activation on the dataset is {highest_activation:.2f}. You must predict the activation on a new sentence based off of the provided description. If the description matches the provided sentence, the activation may be closer to {highest_activation:.2f}, while if it does not match the activation will be nearly 0. \nYou MUST respond with ONLY a number and NO OTHER content.'

    predictions = []

    if debug:
        print(system_prompt)

    # Have the model predict activations on each test sentence
    for sentence in test_sentences:
        sentence_string = sentence['sentence_string']
        user_message = f'Please predict the activation on this sentence,'
        if binary_class:
            user_message += f'responding with either "high" or "low".'
        else:
            user_message += f'responding with a number between 0 and {highest_activation:.2f}.'
        user_message += f'\nSentence: "{sentence_string}" \nRemember, the english description of the feature is: "{description}"'
       
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            n=num_completions,
            max_tokens=2 if binary_class else 5
        )
        
        if debug:
            print(sentence_string)

        prediction = 0
        for i in range(num_completions):
            response = completion.choices[i].message.content
            prediction += parse_binary_response(response) if binary_class else find_first_number(response)
        prediction /= num_completions
        true = sentence['max_value']
        if binary_class:
            true = 1 if sentence['max_value'] == 'high' else 0
        predictions.append((true, prediction))

        if debug:
            print(true, prediction)

    return predictions, extra_data

def predict_wrapper(args):
    feature_index, test_pos, test_neg, show_pos, show_neg, binary_class, neg_type, show_max_token, num_completions, debug, randomize_pos, seed = args
    return predict_activations(feature_index, test_pos=test_pos, test_neg=test_neg, show_pos=show_pos, show_neg=show_neg, binary_class=binary_class, neg_type=neg_type, show_max_token=show_max_token, num_completions=num_completions, debug=debug, randomize_pos=randomize_pos, seed=seed)

def run_experiments(num_features, test_pos=20, test_neg=20, show_pos=0, show_neg=0, binary_class=True, neg_type='others', show_max_token=False, num_completions=1, debug=False, randomize_pos=True, seed=42):
    """
    - num_features: the number of random features to test
    - show_pos and show_neg: the number of positive and negative examples to show GPT3.5, respectively.
    - test_pos and test_neg: the number of positive and negative test examples, respectively.
    - binary_classify specifies whether to predict whether the feature binary activates or does not, otherwise predicts continuous activations
    - neg_type: how we select negative features ('self' selects from the same feature, 'others' selects activating sequences from other features)
    - show_max_token: shows the token with the highest activation and the position in the sentence as which it occurs if True
    - num_completions: how many times to run inference over the model and average the results.
    - debug: prints out the system prompt if True
    - randomize_pos: randomizes the order of the positive examples if True
    - seed is the seed for the random number generator to reproduce the features and examples used to test. Note the GPT3.5 response is not fully reproducible.

    Run the predict_activations function on a set of random feature indices with the above hyperparameters. It saves the results to results/ before returning them.
    """

    timestamp = time.time()
    np.random.seed(seed)
    feature_indices = [int(x) for x in np.random.choice(24000, num_features, replace=False)]

    args = [(feature_index, test_pos, test_neg, show_pos, show_neg, binary_class, neg_type, show_max_token, num_completions, debug, randomize_pos, seed) for feature_index in feature_indices]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        predict_activations_results = list(executor.map(predict_wrapper, args))

    results = {
        'hyperparameters': {
            'test_pos': test_pos,
            'test_neg': test_neg,
            'show_pos': show_pos,
            'show_neg': show_neg,
            'binary_class': binary_class,
            'neg_type': neg_type,
            'show_max_token': show_max_token,
            'num_completions': num_completions,
            'debug': debug,
            'randomize_pos': randomize_pos,
            'seed': seed
        },
        'num_features': num_features,
        'results': [],
        'timestamp': timestamp,
    }
    for i in range(len(feature_indices)):
        
        result_i = {
            'feature_index': feature_indices[i],
            'gpt_predictions': predict_activations_results[i][0],
            **predict_activations_results[i][1]
        }
        results['results'].append(result_i)

    save_json_results(results, f'results/exp_{timestamp}.json')

    return results