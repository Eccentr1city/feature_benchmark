import requests
import json
import pprint

def fetch_and_parse_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def get_activation_data_for_feature(url):
    """
    STRUCTURE OF cleaned_data

    exapmles: list of examples where each example is {'maxValue': int, 'tokens': list, 'values': list}
        tokens and values have same length where values has the activation on each token

    explanations: contains things like
        'author': {'country': 'US', 'name': 'gpt-3.5-turbo'},
        'description': 'phrases related to Q&A sessions or dialogues involving opinions and discussions',
        'layer': '9-res-jb',
        'modelId': 'gpt2-small', 
    """
    parsed_json = fetch_and_parse_json(url)

    cleaned_data = {}
    cleaned_data['explanations'] = parsed_json['explanations']
    examples = []
    for example in parsed_json['activations']:
        examples.append(
            {
                'maxValue': example['maxValue'],
                'tokens': example['tokens'],
                'values': example['values'],
            }
        )
        # assert len(example['tokens']) == len(example['values']), "Error"

    cleaned_data['examples'] = examples

    return cleaned_data

url = "https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/0"
data = get_activation_data_for_feature(url)

pprint.pprint(data)
print(len(data['examples'])) ## Has 76 examples