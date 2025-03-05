import numpy as np
import cupy as cp
from typing import Union, Optional
import neurons as nu

GenArray = Union[cp.ndarray, np.ndarray]  # used repeatedly throughout code


class SynapticConnection:
    def __init__(self, connection_weights: GenArray, mask: GenArray, is_cuda: bool, can_update: bool = True):
        assert connection_weights.shape == mask.shape
        if is_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.connection = connection_weights
        self.mask = mask
        self.can_update = can_update

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

    def __matmul__(self, other):
        connection_matrix = self.mask * self.connection
        return connection_matrix @ other


class IzhNet:
    """
    Base class for Izhikevich Networks
    """

    def __init__(self):
        # self.parameters = []
        self.firing_populations = {}
        self.firing_rates = {}
        self.neural_outputs = {}
        self.population_connections = {}
        self.device = 'cpu'

    def __call__(self, **kwargs):
        self.step(**kwargs)

    def step(self, **kwargs):
        '''

        Args:
            **kwargs:

        Returns:

        '''
        if self.device is 'cpu':
            dev = np
        else:
            dev = cp
        for name, pop in self.firing_populations.items():
            if name in kwargs:
                external_inputs = kwargs[name]
            else:
                external_inputs = []
            if not pop.is_input:
                excitatory_input = dev.zeros(len(pop))
                inhibitory_input = dev.zeros(len(pop))
                for (presynaptic_name, postsynaptic_name), cnxn in self.population_connections.items():
                    if postsynaptic_name == name:
                        # breakpoint()
                        inhibitory = self.firing_populations[presynaptic_name].inhibitory
                        excitatory_input += cnxn @ (self.neural_outputs[name] * dev.logical_not(inhibitory))
                        inhibitory_input += cnxn @ (self.neural_outputs[name] * inhibitory)
                try:
                    pop(excitatory_input, inhibitory_input, *external_inputs)
                except TypeError as e:
                    raise TypeError(f'Invalid input supplied to neuron cluster {name}') from e

            else:
                try:
                    pop(*external_inputs)
                except TypeError as e:
                    raise TypeError(f'Invalid input supplied to neuron cluster {name}') from e

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

    def calc_firing_rates(self):
        for pop in self.firing_rates.values():
            self.firing_rates[pop] = self.firing_populations[pop].firing_ratio()


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
        self.quiescent_rate = .5
        self.active_rate = .5

        if is_cuda:
            dev = cp
        else:
            dev = np
        out_networks = [SimpleNetwork(n_e, n_i, is_cuda, is_conductive, p_mask, f'output_pop_{i}')
                        for i in range(2)]
        in_networks_0 = [nu.SimpleInput(n_e, n_i, is_cuda, is_conductive) for i in range(n_inputs)]
        in_networks_1 = [nu.SimpleInput(n_e, n_i, is_cuda, is_conductive) for i in range(n_inputs)]
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
        for i, input_network in enumerate(in_networks_0):
            self.firing_populations[f'input_{i}_0'] = input_network
            in_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                         dev.random.rand(n_total, n_total) > p_mask, is_cuda)
            self.population_connections[(f'input_{i}_0', 'hidden_pop')] = in_cnxn
            self.neural_outputs[f'input_{i}_0'] = input_network.get_output()

        for i, input_network in enumerate(in_networks_1):
            self.firing_populations[f'input_{i}_1'] = input_network
            in_cnxn = SynapticConnection(abs(dev.random.randn(n_total, n_total)),
                                         dev.random.rand(n_total, n_total) > p_mask, is_cuda)
            self.population_connections[(f'input_{i}_1', 'hidden_pop')] = in_cnxn
            self.neural_outputs[f'input_{i}_1'] = input_network.get_output()

        self.firing_rates = {}
        for pop in self.firing_populations.keys():
            self.firing_rates[pop] = 0.0



    def __call__(self, boolean_input: nu.GenArray, target: nu.GenArray):
        pop_inputs = {}
        for i, boolean in enumerate(boolean_input):
            if boolean == 0:
                pop_inputs[f'input_{i}_0'] = self.quiescent_rate
                pop_inputs[f'input_{i}_1'] = self.active_rate
            else:
                pop_inputs[f'input_{i}_0'] = self.active_rate
                pop_inputs[f'input_{i}_1'] = self.quiescent_rate
        if target == 0:
            pop_inputs[f'teacher_0'] = self.quiescent_rate
            pop_inputs[f'teacher_1'] = self.active_rate
        else:
            pop_inputs[f'teacher_0'] = self.active_rate
            pop_inputs[f'teacher_1'] = self.quiescent_rate

        self.step(**pop_inputs)

    # TODO: set firing rates

    # TODO: burn-in period

    # TODO: read outputs









