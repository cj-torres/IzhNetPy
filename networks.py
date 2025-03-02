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
        # self.parameters = []
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
    def __init__(self, num_excitatory: int, num_inhibitory: int, is_cuda: bool, conductive: bool, p_mask: float = 0,
                 name: str = 'pop'):
        super().__init__()
        if is_cuda:
            dev = cp
        else:
            dev = np
        pop_params = nu.SimpleExcitatoryParams(num_excitatory, is_cuda) + \
            nu.SimpleInhibitoryParams(num_inhibitory, is_cuda)
        pop = nu.IzhPopulation(pop_params, conductive)
        total_num = num_inhibitory + num_excitatory
        synaptic_cnxn = SynapticConnection(abs(dev.random.randn(total_num, total_num)),
                                           dev.random.rand(total_num, total_num) > p_mask, is_cuda)
        self.add_population(pop, name)
        self.add_connection((name, name), synaptic_cnxn)
        self.name = name


class BoolNet(IzhNet):
    def __init__(self, n_inputs: int, n_e: int, n_i: int, is_cuda: bool, is_conductive: bool, p_mask: float = 0):
        super().__init__()
        if is_cuda:
            dev = cp
        else:
            dev = np
        out_networks = [SimpleNetwork(n_e, n_i, is_cuda, is_conductive, p_mask, f'output_pop_{i}')
                        for i in range(2)]
        in_networks = [nu.SimpleInput(n_e, n_i, is_cuda, is_conductive) for i in range(n_inputs)]
        teacher_networks = [nu.SimpleInput(n_e, n_i, is_cuda, is_conductive) for i in range(2)]
        hidden_network = SimpleNetwork(n_e, n_i, is_cuda, is_conductive, p_mask, f'hidden_pop')

        # add hidden network info
        self.firing_populations.update(hidden_network.firing_populations)
        self.population_connections.update(hidden_network.population_connections)
        self.neural_outputs.update(hidden_network.neural_outputs)

        # add output network info, plus connections between it and teachers/hidden layer
        n_total = n_e + n_i
        for i, (network, teacher) in enumerate(zip(out_networks, teacher_networks)):
            self.firing_populations.update(network.firing_populations)
            self.firing_populations[f'teacher_{i}'] = teacher

            self.population_connections.update(network.population_connections)

            teacher_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                              dev.random.rand(n_total, n_total) > p_mask, is_cuda)
            self.population_connections[(f'teacher_{i}', network.name)] = teacher_cnxn

            self.neural_outputs.update(network.neural_outputs)
            self.neural_outputs[f'teacher_{i}'] = teacher.get_output()

            hidden_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                             dev.random.rand(n_total, n_total) > p_mask, is_cuda)
            self.population_connections[('hidden_pop', network.name)] = hidden_cnxn
            for other_network in out_networks:
                if network.name != other_network.name:
                    # generate new connection
                    # may need to make this explicitly inhibitory in the future
                    new_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                                  dev.random.rand(n_total, n_total) > p_mask, is_cuda)
                    self.population_connections[(network.name, other_network.name)] = new_cnxn

        # add input network information
        for i, input_network in enumerate(in_networks):
            self.firing_populations[f'input_{i}'] = input_network
            in_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                         dev.random.rand(n_total, n_total) > p_mask, is_cuda)
            self.population_connections[(f'input_{i}', 'hidden_pop')] = in_cnxn
            self.neural_outputs[f'input_{i}'] = input_network.get_output()








