import networks as n
import learning as l
import plotting as pl
import cupy as cp
import numpy as np


def simple_network_test():
    network = n.SimpleNetwork(800, 200, True, True)
    fired = []
    for t in range(1000):
        network(cp.concatenate([5*cp.random.randn(800), 2*cp.random.randn(200)]))
        fired.append(network.fired)
    pl.plot_raster(cp.asnumpy(cp.stack(fired, axis=0)), 'simple.png')


def two_neuron_depression():
    connections = np.array([[0.0, 0.0], [1.5, 0.0]])
    mask = np.array([[0, 0], [1, 0]])
    network = n.IzhNet(n.RSParams(2, False), connections, True, mask)
    voltage1 = []
    voltage2 = []
    current1 = []
    current2 = []
    d1 = []
    d2 = []
    stdp = l.SimpleSTDP(network, .004, -.004, 15, 20, max_e=2.0)
    for t in range(2000):
        if 25 < t % 200 < 30:
            input_v = np.array([0.0, 12.0])

        elif 35 < t % 100 < 40:
            input_v = np.array([12.0, 0.0])

        else:
            input_v = np.array([0.0, 0])

        network(input_v)
        stdp.update_weights()
        voltage1.append(network.v[0])
        voltage2.append(network.v[1])
        current1.append(network.synaptic_current()[0])
        current2.append(network.synaptic_current()[1])
        d1.append(stdp.d_connections[0, 1])
        d2.append(stdp.d_connections[1, 0])

    pl.plot_neuron_voltages(voltage1, voltage2, "two_neuron_depression.png")
    pl.plot_neuron_voltages(current1, current2, 'two_neuron_current.png')
    pl.plot_neuron_voltages(d1, d2, 'two_neuron_derivative.png')

    print(network.connections)

def two_neuron_facilitation():
    raise NotImplemented
