import requests
import json
import pprint

def fetch_and_parse_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

url = "https://www.neuronpedia.org/api/feature/gpt2-small/9-res-jb/0"
parsed_json = fetch_and_parse_json(url)
# pprint.pprint(parsed_json)

# print(parsed_json.keys())


## activations -> maxValue, tokens, values
## explanations

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

pprint.pprint(cleaned_data['explanations'])

### STRUCTURE OF cleaned_data

### exapmles: list of examples where each example is {'maxValue': int, 'tokens': list, 'values': list}
    ### tokens and values have same length where values has the activation on each token

### explanations: contains things like
#   'author': {'country': 'US', 'name': 'gpt-3.5-turbo'},
#   'description': 'phrases related to Q&A sessions or dialogues involving opinions and discussions',
#   'layer': '9-res-jb',
#   'modelId': 'gpt2-small',
    