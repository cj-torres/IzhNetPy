import networks as n
import learning as l
import plotting as pl
import cupy as cp
import numpy as np
import networks as nw
import neurons as nu


def simple_network_test():
    net = nw.SimpleNetwork(num_excitatory=800, num_inhibitory=200,
                           is_cuda=False, conductive=False, name='test')
    loc = np.concatenate([np.full(800, 5), np.full(200, 2)])
    net.add_population(nu.GaussianPopulation(np.zeros(1000), loc, False), 'noise')
    cnxn = nw.SynapticConnection(np.eye(1000), np.eye(1000), False)
    net.add_connection(('noise', 'test'), cnxn)
    fired = []
    for t in range(1000):
        net()
        fired.append(net.firing_populations['test'].fired)
    pl.plot_raster(np.stack(fired, axis=0), 'simple.png')


# def two_neuron_depression():
#     connections = np.array([[0.0, 0.0], [1.5, 0.0]])
#     mask = np.array([[0, 0], [1, 0]])
#     network = n.IzhNet(n.RSParams(2, False), connections, True, mask)
#     voltage1 = []
#     voltage2 = []
#     current1 = []
#     current2 = []
#     d1 = []
#     d2 = []
#     stdp = l.SimpleSTDP(network, .004, -.004, 15, 20, max_e=2.0)
#     for t in range(2000):
#         if 25 < t % 200 < 30:
#             input_v = np.array([0.0, 12.0])
#
#         elif 35 < t % 100 < 40:
#             input_v = np.array([12.0, 0.0])
#
#         else:
#             input_v = np.array([0.0, 0])
#
#         network(input_v)
#         stdp.update_weights()
#         voltage1.append(network.v[0])
#         voltage2.append(network.v[1])
#         current1.append(network.synaptic_current()[0])
#         current2.append(network.synaptic_current()[1])
#         d1.append(stdp.d_connections[0, 1])
#         d2.append(stdp.d_connections[1, 0])
#
#     pl.plot_neuron_voltages(voltage1, voltage2, "two_neuron_depression.png")
#     pl.plot_neuron_voltages(current1, current2, 'two_neuron_current.png')
#     pl.plot_neuron_voltages(d1, d2, 'two_neuron_derivative.png')
#
#     print(network.connections)
#
# def two_neuron_facilitation():
#     raise NotImplemented
