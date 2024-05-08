import numpy as np
from openai import OpenAI
import re
import pprint
from getting_examples import get_activation_data_for_feature

def find_first_number(text):
    # Return the first number in a string
    match = re.search(r'\b\d+(\.\d+)?', text)
    return float(match.group(0)) if match else None

client = OpenAI()

def predict_activations(feature_index, test_number=20, show_examples=0):
    # Get and parse JSON data from the url corresponding to the requested feature
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/{feature_index}"
    data = get_activation_data_for_feature(url)
    explanation = data['explanations'][0]['description']

    assert (len(data['examples']) >= (test_number + show_examples))

    # Randomly select some sentences to use as examples and test data
    random_indices = np.random.choice(len(data['examples']), size=test_number + show_examples, replace=False)
    sentences = [{'sentence_string': ''.join(data['examples'][i]['tokens']), 'activation':  data['examples'][i]['maxValue'], 'max_index': data['examples'][i]['maxValueTokenIndex']} for i in random_indices]
    example_sentences = sentences[:show_examples]
    test_sentences = sentences[show_examples:] 

    highest_activation = data['examples'][0]['maxValue']

    # Create a system prompt dependning on how many example sentences are provided
    system_prompt = f'You are evaluating an english description of an autoencoder feature. The description should correspond to setences which result in high activation. The english description of the feature is: "{explanation}"\n'

    if show_examples:
        system_prompt += 'Here are 20 examples of sentences and their corresponding activations:\n '
        for sentence in example_sentences:
            sentence_string = sentence['sentence_string']
            activation = sentence['activation']
            system_prompt += f'Example: "{sentence_string}", Activation: {activation}\n'
        system_prompt += 'Use the provided samples and the provided description to predict the activation on a new sentence.'

    else:
        system_prompt += f'The value of the highest activation on the dataset is {highest_activation}. You must predict the activation on a new sentence based off of the provided description – if the description matches the provided sentence, the activation may be closer to {highest_activation}, while if it does not match the activation will be nearly 0.'

    system_prompt += '\nYou MUST respond with ONLY a number and NO OTHER content.'

    predictions = []

    # Have the model predict activations on each test sentence
    for sentence in test_sentences:
        sentence_string = sentence['sentence_string']
        user_message = f'Please predict the activation on this sentence, responding with a number between 0 and {highest_activation}.\n\nSentence: "{sentence_string}"'

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        predicted = find_first_number(completion.choices[0].message.content)
        # (true, pred)
        predictions.append((sentence['activation'], predicted))

    return predictions