import numpy as np
import cupy as cp
from typing import Union, Optional
import neurons as nu

GenArray = Union[cp.ndarray, np.ndarray]  # used repeatedly throughout code


class SynapticConnection:
    def __init__(self, connection_weights: GenArray, mask: GenArray, is_cuda: bool):
        assert connection_weights.shape == mask.shape
        if is_cuda:
            dev = cp
            self.device = "cuda"
        else:
            dev = np
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

    def __call__(self, input_voltages):
        self.step(input_voltages)

    def step(self, input_voltages):
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
            if is_cuda:
                self.population_connections[group_pairs].is_cuda(True)
            else:
                self.population_connections[group_pairs].is_cuda(False)

    def add_population(self, firing_pop: nu.NeuronPopulation, pop_name: str):
        self.firing_populations[pop_name] = firing_pop
        self.neural_outputs[pop_name] = firing_pop.get_output()

    def add_connection(self, group_pairs: tuple[str], connection: SynapticConnection):
        assert all(group in self.firing_populations for group in group_pairs)
        self.population_connections[group_pairs] = connection



class RecurrentIzhNet(IzhNet):
    """
    An Izhikevich network
    """
    def __init__(self, pop: nu.IzhPopulation, connections: GenArray, conductive: bool, mask: Optional[GenArray] = None):
        if type(params.a) == cp.ndarray:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        super().__init__()

        assert type(params.a) == type(connections)

        # standard params
        self.a = params.a
        self.parameters.append('a')
        self.b = params.b
        self.parameters.append('b')
        self.c = params.c
        self.parameters.append('c')
        self.d = params.d
        self.parameters.append('d')

        self.inhibitory = params.inhibitory
        self.parameters.append('inhibitory')

        # conductance params
        self.U = params.U
        self.parameters.append('U')
        self.F = params.F
        self.parameters.append('F')
        self.D = params.D
        self.parameters.append('D')

        self.tau_a = tau_a
        self.parameters.append('tau_a')
        self.tau_b = tau_b
        self.parameters.append('tau_b')
        self.tau_c = tau_c
        self.parameters.append('tau_c')
        self.tau_d = tau_d
        self.parameters.append('tau_d')
        self.connections = connections
        self.parameters.append('connections')
        self.conductive = conductive

        if self.device == 'cuda':
            dev = cp
        else:
            dev = np

        if mask is None:
            self.mask = dev.ones_like(self.connections)
        else:
            self.mask = mask

        self.parameters.append('mask')
        self.connections *= self.connections * self.mask

        self.fired = dev.zeros_like(self.a).astype(bool)
        self.parameters.append('fired')
        self.v = dev.full_like(self.a, -65.0)  # starting voltage
        self.parameters.append('v')
        self.u = self.b * self.v
        self.parameters.append('u')

        # conductance variables
        self.g_a = dev.zeros_like(self.a)
        self.parameters.append('g_a')
        self.g_b = dev.zeros_like(self.a)
        self.parameters.append('g_b')
        self.g_c = dev.zeros_like(self.a)
        self.parameters.append('g_c')
        self.g_d = dev.zeros_like(self.a)
        self.parameters.append('g_d')
        self.R = dev.zeros_like(self.a)
        self.parameters.append('R')
        self.w = dev.zeros_like(self.a)
        self.parameters.append('w')

    def step(self, input_voltages: GenArray, inhibitory: GenArray):
        assert isinstance(input_voltages, type(self.v))
        self.v = self.v
        self.fired = self.v >= 30
        self.v[self.fired] = self.c[self.fired]
        self.u[self.fired] = self.u[self.fired] + self.d[self.fired]

        if self.conductive:
            input_voltages = input_voltages - self.synaptic_current()
        else:
            input_voltages = input_voltages + self.connections @ self.fired

        self.v = self.v + .5 * (.04 * self.v ** 2 + 5 * self.v + 140 - self.u + input_voltages)
        self.v = self.v.clip(max=30)
        self.v = self.v + .5 * (.04 * self.v ** 2 + 5 * self.v + 140 - self.u + input_voltages)
        self.v = self.v.clip(max=30)
        self.u = self.u + self.a * (self.b * self.v - self.u)

        if self.conductive:
            self.conductance_update(input_voltages, inhibitory)
            self.facilitation_update()

    def __call__(self, input_voltages: np.ndarray, inhibitory: GenArray):
        self.step(input_voltages, inhibitory)

    # def is_cuda(self, is_cuda: bool):
    #     """
    #     Switches device for network
    #     :param is_cuda: whether network is on cuda device
    #     :return: None
    #     """
    #
    #     if is_cuda:
    #         self.device = 'cuda'
    #         self.a = cp.array(self.a)
    #         self.b = cp.array(self.b)
    #         self.c = cp.array(self.c)
    #         self.d = cp.array(self.d)
    #         self.fired = cp.array(self.fired)
    #         self.inhibitory = cp.array(self.inhibitory)
    #         self.v = cp.array(self.v)
    #         self.u = cp.array(self.u)
    #         self.U = cp.array(self.U)
    #         self.F = cp.array(self.F)
    #         self.D = cp.array(self.D)
    #         self.connections = cp.array(self.connections)
    #         self.R = cp.array(self.R)
    #         self.w = cp.array(self.w)
    #         self.g_a = cp.array(self.g_a)
    #         self.g_b = cp.array(self.g_b)
    #         self.g_c = cp.array(self.g_c)
    #         self.g_d = cp.array(self.g_d)
    #     else:
    #         self.device = 'cpu'
    #         self.a = cp.asnumpy(self.a)
    #         self.b = cp.asnumpy(self.b)
    #         self.c = cp.asnumpy(self.c)
    #         self.d = cp.asnumpy(self.d)
    #         self.fired = cp.asnumpy(self.fired)
    #         self.inhibitory = cp.asnumpy(self.inhibitory)
    #         self.v = cp.asnumpy(self.v)
    #         self.u = cp.asnumpy(self.u)
    #         self.U = cp.asnumpy(self.U)
    #         self.F = cp.asnumpy(self.F)
    #         self.D = cp.asnumpy(self.D)
    #         self.connections = cp.asnumpy(self.connections)
    #         self.R = cp.asnumpy(self.R)
    #         self.w = cp.asnumpy(self.w)
    #         self.g_a = cp.asnumpy(self.g_a)
    #         self.g_b = cp.asnumpy(self.g_b)
    #         self.g_c = cp.asnumpy(self.g_c)
    #         self.g_d = cp.asnumpy(self.g_d)

    # Conductance functions

    def synaptic_current(self):
        """
        :return:
        """
        a_term = self.g_a * self.v
        b_term = self.g_b * (((self.v + 80) / 60)**2) / (1 + (((self.v + 80) / 60)**2)) * self.v
        c_term = self.g_c*(self.v+70)
        d_term = self.g_d*(self.v+90)
        return a_term + b_term + c_term + d_term

    def conductance_update(self, excitatory_input: GenArray, inhibitory_input: GenArray):
        """
        :return:
        """

        self.g_a = self.g_a + excitatory_input + self.connections @ (
                self.R * self.w * self.fired * (self.inhibitory == 0)) - self.g_a / self.tau_a
        self.g_b = self.g_b + excitatory_input + self.connections @ (
                self.R * self.w * self.fired * (self.inhibitory == 0)) - self.g_b / self.tau_b
        self.g_c = self.g_c + inhibitory_input + self.connections @ (
                self.R * self.w * self.fired * self.inhibitory) - self.g_c / self.tau_c
        self.g_d = self.g_d + inhibitory_input + self.connections @ (
                self.R * self.w * self.fired * self.inhibitory) - self.g_d / self.tau_d

    def facilitation_update(self):
        """
        :return:
        """
        self.R = self.R + (1-self.R)/self.D - self.R * self.w * self.fired
        self.w = self.w + (self.U - self.w)/self.F + self.U * (1-self.w) * self.fired

    def get_output(self, indices: GenArray):
        return (self.fired*self.w*self.R)[indices], self.inhibitory[indices]


class SimpleNetwork(RecurrentIzhNet):
    def __init__(self, num_excitatory: int, num_inhibitory: int, is_cuda: bool, conductive: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        params = SimpleExcitatoryParams(num_excitatory, is_cuda) + SimpleInhibitoryParams(num_inhibitory, is_cuda)
        total = num_inhibitory + num_excitatory
        connections = dev.concatenate([.5*dev.random.rand(total, num_excitatory),   # Excitatory connections
                                       -1*dev.random.rand(total, num_inhibitory)],  # Inhibitory connections
                                      axis=1)
        super().__init__(params, connections, conductive)


class DecisionNet(IzhNet):
    def __init__(self, n_outputs: int, n_e: int, n_i: int, is_cuda: bool, is_conductive: bool):
        super().__init__()
        self.populations = [SimpleNetwork(n_e, n_i, is_cuda, is_conductive) for _ in range(n_outputs)]
