import matplotlib.pyplot as plt
import numpy as np


### Losses
def mse(data, normalize = False):
    values = ([((elem[0]-elem[1])/(elem[0] if normalize else 1))**2 for elem in data])
    return sum(values)/len(values)

def nll_variant(data, eps = 1e-1):
    values = ([np.log((min(elem) + eps)/(max(elem) + eps)) for elem in data])
    return -sum(values)/len(values)

def l1(data, normalize = True, eps = 0.1):
    values = ([((eps + abs(elem[0]-elem[1]))/((max(elem) if normalize else 1) + eps))  for elem in data])
    return sum(values)/len(values)

### Plots
def plot_mses_cdf(mses):
    # Plotting the Mean Squared Errors (MSE) for each dataset
    mses_sorted = np.sort(mses)
    cdf = np.arange(1, len(mses_sorted)+1) / len(mses_sorted)
    plt.plot(mses_sorted, cdf)
    plt.title('Cumulative Distribution Function of MSEs')
    plt.xlabel('MSE')
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

def analyze_data(all_data):
    mses = [mse(data, normalize = False) for data in all_data]
    nlls = [nll_variant(data) for data in all_data]
    l1s = [l1(data, normalize = True) for data in all_data]

    print('l1s', sorted(l1s))
    plot_probability_distribution(mses, title = "Distribution of MSEs")
    plot_probability_distribution(nlls, title = "Distribution of NLL variant")
    plot_probability_distribution(l1s, title = "Distribution of l1s variant")




