import matplotlib
import matplotlib.pyplot as plt
from networks import *
import numpy as np
import cupy as cp

matplotlib.use('TkAgg')

def plot_raster(data: GenArray, filename: str):
    """
    Plots a raster plot from a 2D numpy array.

    Parameters:
    data (numpy.ndarray): A 2D array with shape (T, neurons) where T is the number of time steps
                          and neurons is the number of neurons. True indicates a firing event.

    Returns:
    matplotlib.figure.Figure: The resulting figure.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    fig, ax = plt.subplots(figsize=(10, 6))
    T, neurons = data.shape

    # Extract spike coordinates
    spike_times, neuron_indices = np.where(data)

    # Plot spikes as black dots
    ax.plot(spike_times, neuron_indices, 'k.', markersize=4)

    ax.set_xlabel("Time steps")
    ax.set_ylabel("Neuron")
    ax.set_title("Raster Plot")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.5, neurons - 0.5)
    ax.invert_yaxis()
    plt.grid(False)
    plt.savefig(filename)


def plot_neuron_voltages(voltage1: GenArray, voltage2: GenArray, filename: str):
    plt.figure(figsize=(10, 8))

    # Plot for Neuron 1
    plt.subplot(2, 1, 1)
    plt.plot(range(len(voltage1)), voltage1, label='Neuron 1', color='blue')
    plt.title("Neuron 1 Voltage Over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()

    # Plot for Neuron 2
    plt.subplot(2, 1, 2)
    plt.plot(range(len(voltage2)), voltage2, label='Neuron 2', color='orange')
    plt.title("Neuron 2 Voltage Over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)