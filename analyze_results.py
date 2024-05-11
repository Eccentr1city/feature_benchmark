import matplotlib.pyplot as plt
import numpy as np
import pprint

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
    plt.bar(bin_centers, counts*np.diff(bin_edges), align='center', width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Probability Distribution of Data')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Metrics
def get_binary_accuracy(binary_preds, plot_cdf=False, plot_distribution=False):
    if isinstance(binary_preds[0], list):
        accuracy = ['N/A' if len(elem) == 0 else sum([1-abs(x[0] - x[1]) for x in elem])/len(elem) for elem in binary_preds]
    else:
        accuracy = [1-abs(x[0] - x[1]) for x in binary_preds]/len(binary_preds)
    
    if plot_cdf:
        plot_cdf_graph(accuracy, title = "CDF of accuracy")
    if plot_distribution:
        plot_probability_distribution(accuracy, bins=np.arange(0, 1, 0.01), title = "Distribution of accuracy")
    
    return accuracy

def get_pos_neg_accuracy(binary_preds):
    pos_accuracy = get_binary_accuracy([[[e[0], e[1]] for e in elem if e[0] == 0.0] for elem in binary_preds])
    neg_accuracy = get_binary_accuracy([[[e[0], e[1]] for e in elem if e[0] == 1.0] for elem in binary_preds])
    return pos_accuracy, neg_accuracy

def get_accuracy_descs(json_data_binary, include_pos_neg=False, display=False):
    binary_preds = [json_data_binary['results'][i]['gpt_predictions'] for i in range(len(json_data_binary['results']))]
    descs = [json_data_binary['results'][i]['description'] for i in range(len(json_data_binary['results']))]
    accuracy = get_binary_accuracy(binary_preds)
    
    if include_pos_neg:
        pos_accuracy, neg_accuracy = get_pos_neg_accuracy(binary_preds)
        accuracy_descs = zip(accuracy, pos_accuracy, neg_accuracy, descs)
    else:
        accuracy_descs = zip(accuracy, descs)

    if display:
        for elem in sorted(accuracy_descs, key=lambda x: x[0]):
            pprint.pprint(elem)
    return sorted(accuracy_descs, key=lambda x: x[0])

    


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

# Old function
def analyze_data(all_data):
    mses = [mse(data, normalize = False) for data in all_data]
    nlls = [nll_variant(data) for data in all_data]
    l1s = [l1(data, normalize = True) for data in all_data]

    print('l1s', sorted(l1s))
    plot_probability_distribution(mses, title = "Distribution of MSEs")
    plot_probability_distribution(nlls, title = "Distribution of NLL variant")
    plot_probability_distribution(l1s, title = "Distribution of l1s variant")