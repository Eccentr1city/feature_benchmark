import json
import os
import numpy as np

def recomputed_change(activation):
    eps = 1e-8
    values = activation['values']
    recomputed = activation['recomputedValues'][1:]
    elem_change = [((re - val) /  (max(val, re) + eps)) for re, val in zip(recomputed, values)]
    if (max(sum(values), sum(recomputed))) == 0:
        mag_ratio = 0
    else:
        mag_ratio = sum(recomputed) / (max(sum(values), sum(recomputed)))
    return elem_change, mag_ratio


def compare_recomputed_values(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # (recomputed - value) / value
    pos_elementwise_percent_change = []
    neg_elementwise_percent_change = []
    # sum(recomputed) / max(sum(values), sum(recomputed))
    pos_magnitude_ratios = []
    neg_magnitude_ratios = []

    # Loop through each entry in the posActivations
    for activation in data['posActivations']:
        elem_change, mag_ratio = recomputed_change(activation)
        pos_elementwise_percent_change.append(elem_change)
        pos_magnitude_ratios.append(mag_ratio)
    for activation in data['negSelfActivations']:
        elem_change, mag_ratio = recomputed_change(activation)
        neg_elementwise_percent_change.append(elem_change)
        neg_magnitude_ratios.append(mag_ratio)

    return pos_elementwise_percent_change, neg_elementwise_percent_change, pos_magnitude_ratios, neg_magnitude_ratios, len(data['posActivations'])


def compare_recomputed_group(directory):
    file_names = []
    mean_pos_ratio = []
    mean_neg_ratio = []
    num_positives = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
        file_names.append(os.path.basename(file_path))
        pos_elementwise_percent_change, neg_elementwise_percent_change, pos_magnitude_ratios, neg_magnitude_ratios, num_pos = compare_recomputed_values(file_path)
        num_positives.append(num_pos)
        if len(pos_magnitude_ratios) > 0:
            mean_pos_ratio.append(np.array(pos_magnitude_ratios).mean())
        else:
            mean_pos_ratio.append(0)
        if len(neg_magnitude_ratios) > 0:
            mean_neg_ratio.append(np.array(neg_magnitude_ratios).mean())
        else:
            mean_neg_ratio.append(0)

    return file_names, mean_pos_ratio, mean_neg_ratio, num_positives
