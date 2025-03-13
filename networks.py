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


class InhibitoryConnection(SynapticConnection):
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_inhibitory: float,
                 can_update: bool, p_mask: float = 0, is_brader: bool = True):
        dev = cp if group_1.device == 'cuda' else np
        random_mask = dev.random.rand(len(group_2), len(group_1)) > p_mask
        mask = random_mask * group_1.inhibitory
        if is_brader:
            connections = dev.full_like(mask, w_inhibitory) * mask
        else:
            connections = dev.random.uniform(low=0, high=w_inhibitory, size=(len(group_2), len(group_1))) * mask
        super().__init__(connections, mask, group_1.device == 'cuda', can_update)


class ExcitatoryConnection(SynapticConnection):
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_high: float, w_low: float,
                 can_update: bool, p_mask: float = 0, is_brader: bool = True):
        dev = cp if group_1.device == 'cuda' else np
        random_mask = dev.random.rand(len(group_2), len(group_1)) > p_mask
        mask = random_mask * dev.logical_not(group_1.inhibitory)
        if is_brader:
            connections = dev.zeros((len(group_2), len(group_1)))
            connection_is_high = dev.random.rand(len(group_2), len(group_1)) > .5
            connections[connection_is_high] = w_high
            connections[dev.logical_not(connection_is_high)] = w_low
        else:
            connections = dev.random.uniform(low=w_low, high=w_high, size=(len(group_2), len(group_1)))
        super().__init__(connections, mask, group_1.device == 'cuda', can_update)


class SimpleConnection(SynapticConnection):
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_high: float, w_low: float,
                 w_inhibitory: float, can_update: bool, p_mask: float = 0, is_brader: bool = True):
        dev = cp if group_1.device == 'cuda' else np
        random_mask = dev.random.rand(len(group_2), len(group_1)) > p_mask
        mask = random_mask
        if is_brader:
            connections = dev.zeros((len(group_2), len(group_1)))
            connection_is_high = dev.random.rand(len(group_2), len(group_1)) > .5
            connections[connection_is_high] = w_high
            connections[dev.logical_not(connection_is_high)] = w_low
            connection_is_inhibitory = group_1.inhibitory
            connections[:, connection_is_inhibitory] = w_inhibitory
        else:
            connections = dev.random.uniform(low=w_low, high=w_high, size=(len(group_2), len(group_1)))
        super().__init__(connections, mask, group_1.device == 'cuda', can_update)



class IzhNet:
    """
    Base class for Izhikevich Networks
    """

    def __init__(self):
        # self.parameters = []
        self.firing_populations = {}
        self.firing_history = {}
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
                external_inputs = [kwargs[name]]
            else:
                external_inputs = []
            if not pop.is_input:
                excitatory_input = dev.zeros(len(pop))
                inhibitory_input = dev.zeros(len(pop))
                for (presynaptic_name, postsynaptic_name), cnxn in self.population_connections.items():
                    if postsynaptic_name == name:
                        inhibitory = self.firing_populations[presynaptic_name].inhibitory
                        excitatory_input += cnxn @ (self.neural_outputs[presynaptic_name] * dev.logical_not(inhibitory))
                        inhibitory_input += cnxn @ (self.neural_outputs[presynaptic_name] * inhibitory)
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
        if self.device is 'cpu':
            dev = np
        else:
            dev = cp

        firing_pop.is_cuda(self.device == 'cuda')

        self.firing_populations[pop_name] = firing_pop
        self.neural_outputs[pop_name] = firing_pop.get_output()
        self.firing_rates[pop_name] = firing_pop.firing_ratio()
        self.firing_history[pop_name] = dev.expand_dims(firing_pop.fired, axis=0)

    def add_connection(self, group_pairs: tuple[str, str], connection: SynapticConnection):
        assert all(group in self.firing_populations for group in group_pairs)
        self.population_connections[group_pairs] = connection

    def add_network(self, other_network: 'IzhNet'):
        for pop_name, pop in other_network.firing_populations.items():
            self.add_population(pop, pop_name)
        for pairs, connection in other_network.population_connections.items():
            self.add_connection(pairs, connection)

    def reset_groups(self):
        for group in self.firing_populations.values():
            group.reset()

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
        excitatory_weights = .5*dev.random.rand(total_num, num_excitatory)
        inhibitory_weights = 2*dev.random.rand(total_num, num_inhibitory)

        synaptic_cnxn = SynapticConnection(dev.concatenate([excitatory_weights, inhibitory_weights], axis=1),
                                           dev.random.rand(total_num, total_num) > p_mask, is_cuda)
        self.add_population(pop, name)
        self.add_connection((name, name), synaptic_cnxn)
        self.name = name


class BraderNet(IzhNet):
    def __init__(self, num_excitatory: int, num_inhibitory: int, w_max: float, w_min: float, w_inh: float,
                 is_cuda: bool, conductive: bool, p_mask: float = 0, name: str = 'pop'):
        super().__init__()
        pop_params = nu.SimpleExcitatoryParams(num_excitatory, is_cuda) + \
            nu.SimpleInhibitoryParams(num_inhibitory, is_cuda)
        pop = nu.IzhPopulation(pop_params, conductive)

        synaptic_cnxn = SimpleConnection(pop, pop, w_max, w_min, w_inh, True, p_mask=p_mask)
        self.add_population(pop, name)
        self.add_connection((name, name), synaptic_cnxn)
        self.name = name


class BoolNet(IzhNet):
    def __init__(self, n_inputs: int, n_e: int, n_i: int, w_max: float, w_min: float, w_inh: float, is_cuda: bool,
                 is_conductive: bool, p_mask: float = 0):
        super().__init__()
        self.n_inputs = n_inputs
        self.quiescent_rate = .5
        self.active_rate = .5
        self.teacher_pops = set()
        self.output_pops = set()

        if is_cuda:
            dev = cp
        else:
            dev = np
        hidden_sz = n_e+n_i
        input_sz = hidden_sz//(n_inputs*2)

        out_networks = [BraderNet(n_e, n_i, w_min, w_max, w_inh, is_cuda, is_conductive, p_mask, f'output_pop_{i}')
                        for i in range(2)]
        in_networks_0 = [nu.SimpleInput(input_sz, 0, is_cuda, is_conductive) for i in range(n_inputs)]
        in_networks_1 = [nu.SimpleInput(input_sz, 0, is_cuda, is_conductive) for i in range(n_inputs)]
        teacher_networks = [nu.SimpleInput(hidden_sz//2, 0, is_cuda, is_conductive) for i in range(2)]
        hidden_network = BraderNet(n_e, n_i, w_min, w_max, w_inh, is_cuda, is_conductive, p_mask, f'hidden_pop')

        # add hidden network info
        self.add_network(hidden_network)
        hidden_group = hidden_network.firing_populations[hidden_network.name]

        # add output network info, plus connections between it and teachers/hidden layer
        for i, (network, teacher) in enumerate(zip(out_networks, teacher_networks)):
            self.add_network(network)
            self.add_population(teacher, f'teacher_{i}')

            out_group = network.firing_populations[network.name]

            teacher_cnxn = ExcitatoryConnection(
                teacher, out_group, w_max, w_min, False
            )

            self.population_connections[(f'teacher_{i}', network.name)] = teacher_cnxn
            self.output_pops.add(network.name)
            self.teacher_pops.add(f'teacher_{i}')

            hidden_cnxn = SimpleConnection(
                hidden_group, out_group, w_max, w_min, w_inh, True, p_mask=p_mask
            )

            self.population_connections[('hidden_pop', network.name)] = hidden_cnxn

        for group_name in self.output_pops:
            for other_group_name in self.output_pops:
                if group_name != other_group_name:
                    pre_group = self.firing_populations[group_name]
                    post_group = self.firing_populations[other_group_name]
                    self.population_connections[(group_name, other_group_name)] = InhibitoryConnection(
                        pre_group, post_group, w_inh, False
                    )

        self.input_pops = set()
        # add input network information
        for i, input_network in enumerate(in_networks_0):
            self.add_population(input_network, f'input_{i}_0')
            in_cnxn = ExcitatoryConnection(
                input_network, hidden_group, w_max, w_min, False
            )
            self.population_connections[(f'input_{i}_0', 'hidden_pop')] = in_cnxn
            self.input_pops.add(f'input_{i}_0')

        for i, input_network in enumerate(in_networks_1):
            self.add_population(input_network, f'input_{i}_1')
            in_cnxn = ExcitatoryConnection(
                input_network, hidden_group, w_max, w_min, False
            )
            self.population_connections[(f'input_{i}_1', 'hidden_pop')] = in_cnxn
            self.input_pops.add(f'input_{i}_1')

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

    def tune_to_rate(
            self,
            target_rate: float,
            populations_to_monitor: set[str],
            steps_checked: int = 5000,  # Check on full second
            max_iters: int = 20,
            tolerance: float = 1e-3
    ):
        """
        Adjust self.active_rate so that the average firing ratio across
        'populations_to_monitor' approaches 'target_rate'.

        :param target_rate: Desired firing ratio, e.g. 0.15 for 15% firing
        :param populations_to_monitor: Set of population names whose firing rates we average
        :param steps_checked: Number of simulation steps per tuning iteration
        :param max_iters: Maximum number of iterations to attempt
        :param tolerance: Stop when |avg_rate - target_rate| < tolerance
        """
        low = 0.0
        high = 1.0
        rate = .5

        assert(max_iters >= 1)

        for iteration in range(max_iters):
            # Pick midpoint for active_rate
            mid = (low + high) / 2.0
            rate = mid

            ratios = []

            # Run the network for 'steps_checked' steps to measure new firing ratio
            for _ in range(steps_checked):
                # Provide default/dummy inputs to the step function using active_rate
                self.step(**{pop_name: rate for pop_name in self.input_pops.union(self.teacher_pops)})

                avg_rate = 0.0
                for pop_name in populations_to_monitor:
                    avg_rate += self.firing_populations[pop_name].firing_ratio()
                avg_rate /= len(populations_to_monitor)

                ratios.append(avg_rate)

            # Calculate average firing ratio over all steps
            average = sum(ratios) / len(ratios)
            diff = average - target_rate
            print(f"[Tune Active] Iter={iteration}, a_rate={mid:.4f}, avg_rate={average:.4f}")

            # Check if we've reached the target within tolerance
            if abs(diff) < tolerance:
                print(f"[Tune Active] Converged: avg_rate={average:.4f}, target={target_rate:.4f}")
                break

            # Adjust binary-search bounds
            if average < target_rate:
                # If rate is too low, move lower bound up
                low = mid
            else:
                # If rate is too high, move upper bound down
                high = mid

            # Stop if bounds are extremely close
            if (high - low) < 1e-6:
                print("[Tune Active] Bounds converged closely.")
                break

            self.reset_groups()

        return rate

    def tune_quiescent_rate(self, target_rate: float, populations_to_monitor: set[str],
                            steps_checked: int = 5000,  # Check on full second
                            max_iters: int = 20, tolerance: float = 1e-3):

        self.quiescent_rate = self.tune_to_rate(target_rate, populations_to_monitor, steps_checked, max_iters, tolerance)

    def tune_active_rate(self, target_rate: float, populations_to_monitor: set[str],
                         steps_checked: int = 5000,  # Check on full second
                         max_iters: int = 20, tolerance: float = 1e-3):

        self.active_rate = self.tune_to_rate(target_rate, populations_to_monitor, steps_checked, max_iters, tolerance)


    # TODO: burn-in period

    # TODO: read outputs





def brader_init(connection: SynapticConnection, w_high: float, w_low: float,
                 w_inhibitory: float, p_connected_e: float, p_connected_i: float):
    dev = cp if connection.device == 'cuda' else np
    connection.connection[:, :] = 0.0
    excitatory_cnxn = dev.random.rand(len(connection.group_2), len(connection.group_1)) > p_connected_e
    excitatory_cnxn = dev.logical_and(excitatory_cnxn, dev.logical_not(connection.group_1.inhibitory))
    inhibitory_cnxn = dev.random.rand(len(connection.group_2), len(connection.group_1)) > p_connected_e
    inhibitory_cnxn = dev.logical_and(inhibitory_cnxn, connection.group_1.inhibitory)
    connection.connection[excitatory_cnxn] = w_high
    connection.connection[dev.logical_not(excitatory_cnxn)] = w_low
    connection.connection[inhibitory_cnxn] = w_inhibitory


def zero_init(connection: SynapticConnection):
    connection.connection[:, :] = 0.0



