from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pprint
from tqdm import tqdm


# Take experiment file, compute the loss change when using predicted activations,
# and add it back to the file
# Get model's loss on strings
def get_experiment_losses(experiment_file, model, sae):
    with open(experiment_file, 'r') as file:
        data = json.load(file)
    
    for feature in tqdm(data['results']):
        feat_id = feature['feature_index']
        for example in feature['gpt_predictions']:
            sentence_string = [example['sentence_string']]

            _, inner_acts, _ = get_sae_activations(model, sae, sentence_string)
            
            # Model loss
            regular_losses = get_vanilla_loss(model, sae, sentence_string)
            example['regular_losses'] = regular_losses[0]

            # Loss using SAE reconstructed activations
            sae_losses = get_vanilla_loss(model, sae, sentence_string, with_sae_replacement=True)
            example['sae_losses'] = sae_losses[0]

            # Loss with all features ablated
            precomputed_zeros = [[[0.0] * len(l) for l in seq] for seq in inner_acts]
            zeros_losses = get_recons_loss_from_predicted_values(model, sae, sentence_string, precomputed_zeros)
            example['ablated_sae_losses'] = zeros_losses[0]

            # Loss with specific feature ablated
            ablated_inner_acts = replace_feature_activation(inner_acts, feat_id, 0)
            ablated_feature_losses = get_recons_loss_from_predicted_values(model, sae, sentence_string, ablated_inner_acts)
            example['ablated_feature_losses'] = ablated_feature_losses[0]

            # Loss using predicted activations
            
            replacements = [example['prediction']]
            assert len(inner_acts[0]) == len(replacements[0]), f'IS this the problem?? {len(inner_acts[0])}, {len(replacements[0])}'
            replaced_inner_acts = replace_sequence_feature_activation(inner_acts, feat_id, replacements)
            replaced_sae_losses = get_recons_loss_from_predicted_values(model, sae, sentence_string, replaced_inner_acts)
            example['predicted_feature_losses'] = replaced_sae_losses[0]

    with open(experiment_file, 'w') as file:
        json.dump(data, file)


def analyze_loss_change(experiment_file):
    # make some plots maybe?
    # Average total loss change across sequece per feature
    # Average loss change per token per feature
    return

# === OLD ===
# Most of the stuff below here is not needed anymore

# Plots
def plot_cdf_graph(data, title = "Default Title"):
    # Plotting the Mean Squared Errors (MSE) for each dataset
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    plt.plot(data_sorted, cdf)
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.show()

def plot_probability_distribution(data, bins='auto', density=True, title = "Default Title"):
    """
    Plots the probability distribution of the given data using a histogram.

    Parameters:
    - data (list or numpy array): The floating point numbers whose distribution you want to plot.
    - bins (int, sequence or str, optional): The method for calculating histogram bins. Default is 'auto'.
    - density (bool, optional): If True, the histogram is normalized to form a probability density,
                                i.e., the area under the histogram will sum to 1. Default is True.
    """
    # Calculate the histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=density)

    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, counts*np.diff(bin_edges), align='center', width=np.diff(bin_edges), alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Probability Distribution of Data')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Metrics
def get_binary_accuracy(binary_preds, plot_cdf=False, plot_distribution=False):
    # Convert binary_preds to a numpy array
    binary_preds = np.array(binary_preds)
    
    if binary_preds.ndim == 3:  # Check if binary_preds is a list of lists of pairs
        # Calculate accuracy for each list of pairs
        accuracies = np.array([np.nan if len(elem) == 0 else np.mean(1 - np.abs(elem[:, 0] - elem[:, 1])) for elem in binary_preds])
    else:
        # Calculate the overall accuracy
        accuracies = np.mean(1 - np.abs(binary_preds[:, 0] - binary_preds[:, 1]))
    
    if plot_cdf:
        plot_cdf_graph(accuracies, title="CDF of accuracy")
    if plot_distribution:
        plot_probability_distribution(accuracies, bins=np.arange(0, 1.01, 0.01), title="Distribution of accuracy")
    
    return accuracies

# TODO: EASY MODE for the model (no magnitude or location prediction):
# For each binary prediction:
# If true positive or true negative, leave activations unchanged. 
# If false positive, mean-ablate the feature (mean among pos examples); if false negative, zero-ablate the feature
# Measure the change in loss
def easy_binary_pred_loss(binary_preds):
    return


def get_pos_neg_accuracy(binary_preds):

    binary_preds = np.array(binary_preds)
    
    # Compute positive and negative accuracies
    pos_accuracy = get_binary_accuracy([elem[np.where(elem[:, 0] == 1.0)] for elem in binary_preds])
    neg_accuracy = get_binary_accuracy([elem[np.where(elem[:, 0] == 0.0)] for elem in binary_preds])
    
    return pos_accuracy, neg_accuracy


def get_accuracy_descs(json_data_binary, include_pos_neg=False, display=False):
    binary_preds = [json_data_binary['results'][i]['gpt_predictions'] for i in range(len(json_data_binary['results']))]
    descs = [json_data_binary['results'][i]['description'] for i in range(len(json_data_binary['results']))]
    accuracy = get_binary_accuracy(binary_preds)
    
    if include_pos_neg:
        pos_accuracy, neg_accuracy = get_pos_neg_accuracy(binary_preds)
        accuracy_descs = list(zip(accuracy, pos_accuracy, neg_accuracy, descs))
    else:
        accuracy_descs = list(zip(accuracy, descs))

    if display:
        for elem in sorted(accuracy_descs, key=lambda x: x[0]):
            pprint.pprint(elem)

    return accuracy_descs


# Losses
def mse(data, normalize = False):
    values = ([((elem[0]-elem[1])/(elem[0] if normalize else 1))**2 for elem in data])
    return sum(values)/len(values)

def nll_variant(data, eps = 1e-1):
    values = ([np.log((min(elem) + eps)/(max(elem) + eps)) for elem in data])
    return -sum(values)/len(values)

def l1(data, normalize = True, eps = 0.1):
    values = ([((eps + abs(elem[0]-elem[1]))/((max(elem) if normalize else 1) + eps))  for elem in data])
    return sum(values)/len(values)

# def auroc(data):
