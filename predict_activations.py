import concurrent.futures
from functools import partial
from getting_examples import *
import json
from multiprocessing import Pool
import numpy as np
from openai import OpenAI
import os
import pprint
import random
import time
from utils import *

client = OpenAI()

def ask_model(system_prompt, user_message, num_completions, binary_class):
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            n=num_completions,
            max_tokens=2 if binary_class else 5
        )
    # print('tokens:', completion.usage.prompt_tokens)
    return completion

def predict_activations(feature_index, layer, basis, test_pos=20, test_neg=20, show_pos=0, show_neg=0, binary_class=True, neg_type='others', show_max_token=False, num_completions=1, debug=False, randomize_pos=True, seed=42):
    # Get positive and negative examples of the feature activation
    num_pos = test_pos + show_pos
    num_neg = test_neg + show_neg
    description, pos_examples, neg_examples, highest_activation = get_pos_neg_examples(feature_index, layer, basis, num_pos=num_pos, num_neg=num_neg, neg_type=neg_type, randomize_pos_examples=randomize_pos, seed=seed)

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
        for i, sentence in enumerate(show_sentences):
            sentence_string = sentence['sentence_string']
            activation = sentence['max_value']
            if not binary_class:
                activation = round(sentence['max_value'], 2)
            system_prompt += f'Example: "{sentence_string}", Activation: {activation}'
            if show_max_token and i < show_pos:
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
       
        completion = ask_model(system_prompt, user_message, num_completions, binary_class)
        
        if debug:
            print(sentence_string)

        prediction = 0
        for i in range(num_completions):
            response = completion.choices[i].message.content
            pred = parse_binary_response(response) if binary_class else find_first_number(response)
            if pred is None:
                print('WARNING: Resampling')
                resample = ask_model(system_prompt, user_message, 1, binary_class).choices[0].message.content
                pred = parse_binary_response(resample) if binary_class else find_first_number(resample)
            if pred is None:
                raise Exception(f"No valid model prediction for user_message: {user_message}")
            prediction += pred
        prediction /= num_completions
        true = sentence['max_value']
        if binary_class:
            true = 1 if sentence['max_value'] == 'high' else 0
        predictions.append((true, prediction))

        if debug:
            print(true, prediction)

    return predictions, extra_data

def predict_wrapper(args):
    feature_index, layer, basis, test_pos, test_neg, show_pos, show_neg, binary_class, neg_type, show_max_token, num_completions, debug, randomize_pos, seed = args
    return predict_activations(feature_index, layer, basis, test_pos=test_pos, test_neg=test_neg, show_pos=show_pos, show_neg=show_neg, binary_class=binary_class, neg_type=neg_type, show_max_token=show_max_token, num_completions=num_completions, debug=debug, randomize_pos=randomize_pos, seed=seed)

def run_experiments(num_features, layer, basis, test_pos=20, test_neg=20, show_pos=0, show_neg=0, binary_class=True, neg_type='others', show_max_token=False, num_completions=1, debug=False, randomize_pos=True, seed=42, save_location=''):
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
    - save_location: the location to save the results to in results/ if not ''

    Run the predict_activations function on a set of random feature indices with the above hyperparameters. It saves the results to results/ before returning them.
    """

    assert layer in autoencoder_layers, f"Invalid layer: {layer} not in {autoencoder_layers}"
    assert basis in autoencoder_bases, f"Invalid basis: {basis} not in {autoencoder_bases}"

    timestamp = time.time()
    np.random.seed(seed)
    n_layers = num_layers(basis)
    feature_indices = []
    while len(feature_indices) < num_features:
        # f_id = int(np.random.choice(num_layers, 1, replace=False))
        f_id = random.choice(range(n_layers))
        if f_id not in feature_indices and features_exist(layer, basis, f_id, depth=test_pos+show_pos):
            feature_indices.append(f_id)

    args = [(feature_index, layer, basis, test_pos, test_neg, show_pos, show_neg, binary_class, neg_type, show_max_token, num_completions, debug, randomize_pos, seed) for feature_index in feature_indices]
    
    predict_activations_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, arg in enumerate(args):
            time.sleep(4) # Change this number if doing significantly more than 10 test features per feature_id
            future = executor.submit(predict_wrapper, arg)
            futures.append(future)
            print(f"Submitted {i+1} of {num_features} tasks. Been running for {int(time.time() - timestamp)} seconds")
        for future in futures:
            result = future.result()
            predict_activations_results.append(result)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     predict_activations_results = list(executor.map(predict_wrapper, args))

    results = {
        'hyperparameters': {
            'num_features': num_features,
            'layer': layer,
            'basis': basis,
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

    if save_location:
        save_dir = f'results/{save_location}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = 'results'

    save_json_results(results, f'{save_dir}/exp_{timestamp}.json')

    return results

