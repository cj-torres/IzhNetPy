import numpy as np
import cupy as cp
from typing import Union, Optional
import neurons as nu

GenArray = Union[cp.ndarray, np.ndarray]  # used repeatedly throughout code


class SynapticConnection:
    def __init__(self, connection_weights: GenArray, mask: GenArray, is_cuda: bool):
        assert connection_weights.shape == mask.shape
        if is_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.connection = connection_weights
        self.mask = mask

    def remask(self):
        self.connection = self.connection * self.mask

    def is_cuda(self, is_cuda: bool):
        if is_cuda:
            self.device = "cuda"
            if not isinstance(self.connection, cp.ndarray):
                self.connection = cp.asarray(self.connection)
            if not isinstance(self.mask, cp.ndarray):
                self.mask = cp.asarray(self.mask)
        else:
            self.device = "cpu"
            if not isinstance(self.connection, np.ndarray):
                self.connection = cp.asnumpy(self.connection)
            if not isinstance(self.mask, np.ndarray):
                self.mask = cp.asnumpy(self.mask)


class IzhNet:
    """
    Base class for Izhikevich Networks
    """

    def __init__(self):
        self.parameters = []
        self.firing_populations = {}
        self.neural_outputs = {}
        self.population_connections = {}
        self.device = 'cpu'

    def __call__(self):
        self.step()

    def step(self):
        if self.device is 'cpu':
            dev = np
        else:
            dev = cp
        for name, pop in self.firing_populations.items():
            input_voltage = dev.zeros(pop.shape[0])
            for (presynaptic_name, postsynaptic_name), cnxn in self.population_connections.items():
                if postsynaptic_name == name:
                    input_voltage += cnxn @ self.neural_outputs[name]
            pop(input_voltage)
        for name, pop in self.firing_populations.items():
            self.neural_outputs[name] = pop.get_output()

    def is_cuda(self, is_cuda: bool):
        """
        Switches device for network
        :param is_cuda: whether network is on cuda device
        :return: None
        """
        self.device = 'cuda' if is_cuda else 'cpu'
        for group in self.firing_populations.values():
            group.is_cuda(is_cuda)
        for group_pairs in self.population_connections.keys():
            self.population_connections[group_pairs].is_cuda(is_cuda)

    def add_population(self, firing_pop: nu.NeuronPopulation, pop_name: str):
        self.firing_populations[pop_name] = firing_pop
        self.neural_outputs[pop_name] = firing_pop.get_output()

    def add_connection(self, group_pairs: tuple[str, str], connection: SynapticConnection):
        assert all(group in self.firing_populations for group in group_pairs)
        self.population_connections[group_pairs] = connection


class SimpleNetwork(IzhNet):
    def __init__(self, num_excitatory: int, num_inhibitory: int, is_cuda: bool, conductive: bool, p_mask: float = 0):
        super().__init__()
        if is_cuda:
            dev = cp
        else:
            dev = np
        pop_params = nu.SimpleExcitatoryParams(num_excitatory, is_cuda) + \
            nu.SimpleInhibitoryParams(num_inhibitory, is_cuda)
        pop = nu.IzhPopulation(pop_params, conductive)
        total_num = num_inhibitory + num_excitatory
        synaptic_cnxn = SynapticConnection(dev.random.randn(total_num, total_num),
                                           dev.random.rand(total_num, total_num) > p_mask, is_cuda)
        self.add_population(pop, 'pop_0')
        self.add_connection(('pop_0', 'pop_0'), synaptic_cnxn)


class BoolNet(IzhNet):
    def __init__(self, n_outputs: int, n_e: int, n_i: int, is_cuda: bool, is_conductive: bool):
        super().__init__()
        self.populations = [SimpleNetwork(n_e, n_i, is_cuda, is_conductive) for _ in range(n_outputs)]
