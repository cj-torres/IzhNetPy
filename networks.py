"""
This module provides classes for creating and simulating networks of Izhikevich neurons.

The module is organized into two main types of classes:
1. Connection classes (SynapticConnection and subclasses) - These define the synaptic
   connections between neuron populations, with different types of connections having
   different weight initialization and behavior.
2. Network classes (IzhNet and subclasses) - These implement complete neural networks
   composed of neuron populations and connections between them.

The networks use neuron populations from the neurons module, and can be run on either
CPU (using NumPy) or GPU (using CuPy).
"""

import numpy as np
import cupy as cp
from typing import Union, Optional
import neurons as nu

GenArray = Union[cp.ndarray, np.ndarray]  # used repeatedly throughout code


class SynapticConnection:
    """
    Base class for synaptic connections between neuron populations.

    This class represents the synaptic connections between two neuron populations,
    with a weight matrix that determines the strength of the connections. The connections
    can be masked to create specific connectivity patterns, and they can be set to be
    updatable or not (for learning).

    The connection weights are stored as a matrix where each row corresponds to a
    postsynaptic neuron and each column corresponds to a presynaptic neuron. The
    value at position (i, j) represents the strength of the connection from presynaptic
    neuron j to postsynaptic neuron i.
    """
    def __init__(self, connection_weights: GenArray, presynaptic_group: nu.NeuronPopulation,
                 postsynaptic_group: nu.NeuronPopulation, mask: GenArray, is_cuda: bool, can_update: bool = True):
        """
        Initialize a synaptic connection between two neuron populations.

        Args:
            connection_weights: Matrix of synaptic weights
            presynaptic_group: Source neuron population
            postsynaptic_group: Target neuron population
            mask: Binary mask for the connection matrix (1 = connected, 0 = not connected)
            is_cuda: Whether to use GPU acceleration
            can_update: Whether the connection weights can be updated during learning
        """
        assert connection_weights.shape == mask.shape
        assert connection_weights.shape[1] == len(presynaptic_group)
        assert connection_weights.shape[0] == len(postsynaptic_group)
        if is_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.presynaptic_group = presynaptic_group
        self.postsynaptic_group = postsynaptic_group
        self.connection = connection_weights
        self.mask = mask
        self.can_update = can_update

    def remask(self):
        """
        Apply the mask to the connection weights.

        This method multiplies the connection weights by the mask, effectively
        zeroing out connections that are masked (where mask = 0).
        """
        self.connection = self.connection * self.mask

    def is_cuda(self, is_cuda: bool):
        """
        Switch the connection between CPU and GPU.

        This method moves the connection weights and mask to either the CPU (using NumPy)
        or the GPU (using CuPy) based on the is_cuda parameter.

        Args:
            is_cuda: Whether to use GPU acceleration (True) or CPU (False)
        """
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
        """
        Matrix multiplication operator for applying the connection to neural outputs.

        This method applies the masked connection weights to the output of the presynaptic
        neurons to compute the input to the postsynaptic neurons.

        Args:
            other: Output of the presynaptic neurons

        Returns:
            Input to the postsynaptic neurons
        """
        connection_matrix = self.mask * self.connection
        return connection_matrix @ other


class InhibitoryConnection(SynapticConnection):
    """
    A connection that only includes inhibitory synapses.

    This class creates connections from inhibitory neurons in the presynaptic group
    to neurons in the postsynaptic group. Connections from excitatory neurons are
    masked out. This is useful for creating specific inhibitory pathways in a network.
    """
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_inhibitory: float,
                 can_update: bool, p_mask: float = 0, is_brader: bool = True):
        """
        Initialize an inhibitory connection between two neuron populations.

        Args:
            group_1: Presynaptic neuron population
            group_2: Postsynaptic neuron population
            w_inhibitory: Weight for inhibitory connections (typically negative)
            can_update: Whether the connection weights can be updated during learning
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
            is_brader: Whether to use Brader-style initialization (binary weights) or
                       continuous random weights
        """
        dev = cp if group_1.device == 'cuda' else np
        random_mask = dev.random.rand(len(group_2), len(group_1)) > p_mask
        mask = random_mask * group_1.inhibitory
        if is_brader:
            connections = dev.full_like(mask, w_inhibitory) * mask
        else:
            connections = dev.random.uniform(low=0, high=w_inhibitory, size=(len(group_2), len(group_1))) * mask
        super().__init__(connections, group_1, group_2, mask, group_1.device == 'cuda', can_update)


class ExcitatoryConnection(SynapticConnection):
    """
    A connection that only includes excitatory synapses.

    This class creates connections from excitatory neurons in the presynaptic group
    to neurons in the postsynaptic group. Connections from inhibitory neurons are
    masked out. This is useful for creating specific excitatory pathways in a network.
    """
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_high: float, w_low: float,
                 can_update: bool, p_mask: float = 0, is_brader: bool = True):
        """
        Initialize an excitatory connection between two neuron populations.

        Args:
            group_1: Presynaptic neuron population
            group_2: Postsynaptic neuron population
            w_high: High weight value for excitatory connections (typically positive)
            w_low: Low weight value for excitatory connections (typically positive but lower than w_high)
            can_update: Whether the connection weights can be updated during learning
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
            is_brader: Whether to use Brader-style initialization (binary weights) or
                       continuous random weights
        """
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
        super().__init__(connections, group_1, group_2, mask, group_1.device == 'cuda', can_update)


class SimpleConnection(SynapticConnection):
    """
    A connection that includes both excitatory and inhibitory synapses.

    This class creates connections from both excitatory and inhibitory neurons in the
    presynaptic group to neurons in the postsynaptic group. Excitatory connections
    have weights between w_low and w_high, while inhibitory connections have weight
    w_inhibitory. This is useful for creating general connections in a network.
    """
    def __init__(self, group_1: nu.NeuronPopulation, group_2: nu.NeuronPopulation, w_high: float, w_low: float,
                 w_inhibitory: float, can_update: bool, p_mask: float = 0, is_brader: bool = True):
        """
        Initialize a simple connection between two neuron populations.

        Args:
            group_1: Presynaptic neuron population
            group_2: Postsynaptic neuron population
            w_high: High weight value for excitatory connections (typically positive)
            w_low: Low weight value for excitatory connections (typically positive but lower than w_high)
            w_inhibitory: Weight for inhibitory connections (typically negative)
            can_update: Whether the connection weights can be updated during learning
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
            is_brader: Whether to use Brader-style initialization (binary weights) or
                       continuous random weights
        """
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
        super().__init__(connections, group_1, group_2, mask, group_1.device == 'cuda', can_update)



class IzhNet:
    """
    Base class for Izhikevich Networks.

    This class provides the foundation for building networks of Izhikevich neurons.
    It manages collections of neuron populations and the connections between them,
    and provides methods for running the network, adding populations and connections,
    and monitoring network activity.

    The network can be run on either CPU (using NumPy) or GPU (using CuPy), and
    it supports both regular and input-type neuron populations.
    """

    def __init__(self):
        """
        Initialize an empty Izhikevich network.

        Creates empty dictionaries for storing neuron populations, their firing history,
        firing rates, neural outputs, and connections between populations. The network
        is initialized to run on CPU by default.
        """
        # self.parameters = []
        self.firing_populations = {}  # Dictionary of neuron populations
        self.firing_history = {}      # Dictionary of firing history for each population
        self.firing_rates = {}        # Dictionary of firing rates for each population
        self.neural_outputs = {}      # Dictionary of neural outputs for each population
        self.population_connections = {}  # Dictionary of connections between populations
        self.device = 'cpu'           # Device to run the network on (CPU or GPU)

    def __call__(self, **kwargs):
        """
        Call the network's step method.

        This allows the network to be called like a function, which is a convenient
        shorthand for running a single step of the network.

        Args:
            **kwargs: Keyword arguments to pass to the step method
        """
        self.step(**kwargs)

    def step(self, **kwargs):
        """
        Run a single step of the network.

        This method updates all neuron populations in the network based on their
        current state and the inputs they receive from other populations and
        external sources.

        Args:
            **kwargs: External inputs to the network, where keys are population names
                     and values are the inputs to those populations
        """
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
        """
        Add a neuron population to the network.

        This method adds a neuron population to the network with the given name,
        initializes its neural outputs, firing rates, and firing history, and
        ensures it's running on the same device as the network.

        Args:
            firing_pop: The neuron population to add
            pop_name: The name to give the population in the network
        """
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
        """
        Add a connection between two populations in the network.

        This method adds a synaptic connection between two populations in the network.
        The populations are identified by their names in the network.

        Args:
            group_pairs: A tuple of (presynaptic_population_name, postsynaptic_population_name)
            connection: The synaptic connection to add
        """
        assert all(group in self.firing_populations for group in group_pairs)
        self.population_connections[group_pairs] = connection

    def add_network(self, other_network: 'IzhNet'):
        """
        Add another network to this network.

        This method adds all populations and connections from another network to this network.
        This is useful for building hierarchical networks or combining multiple networks.

        Args:
            other_network: The network to add to this network
        """
        for pop_name, pop in other_network.firing_populations.items():
            self.add_population(pop, pop_name)
        for pairs, connection in other_network.population_connections.items():
            self.add_connection(pairs, connection)

    def reset_groups(self):
        """
        Reset all neuron populations in the network.

        This method resets all neuron populations in the network to their initial state.
        This is useful for running multiple trials or experiments with the same network.
        """
        for group in self.firing_populations.values():
            group.reset()

    def calc_firing_rates(self):
        """
        Calculate the firing rates of all populations in the network.

        This method updates the firing_rates dictionary with the current firing rates
        of all populations in the network.
        """
        for pop in self.firing_rates.values():
            self.firing_rates[pop] = self.firing_populations[pop].firing_ratio()


class SimpleNetwork(IzhNet):
    """
    A simple network with a single population of Izhikevich neurons.

    This class creates a network with a single population of Izhikevich neurons,
    which includes both excitatory and inhibitory neurons. The neurons are connected
    with random weights, and the network can be run on either CPU or GPU.

    This is a basic implementation of an Izhikevich network that can be used as a
    starting point for more complex networks.
    """
    def __init__(self, num_excitatory: int, num_inhibitory: int, is_cuda: bool, conductive: bool, p_mask: float = 0,
                 name: str = 'pop'):
        """
        Initialize a simple network with a single population of Izhikevich neurons.

        Args:
            num_excitatory: Number of excitatory neurons in the population
            num_inhibitory: Number of inhibitory neurons in the population
            is_cuda: Whether to use GPU acceleration
            conductive: Whether to use conductance-based synapses
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
            name: Name of the population in the network
        """
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
    """
    A network with a single population of Izhikevich neurons using Brader-style connections.

    This class creates a network with a single population of Izhikevich neurons,
    which includes both excitatory and inhibitory neurons. The neurons are connected
    with Brader-style connections, which use binary weights (either w_min or w_max for
    excitatory connections, and w_inh for inhibitory connections).

    This type of network is based on the model described in Brader et al. (2007),
    which is used for spike-timing-dependent plasticity (STDP) learning.
    """
    def __init__(self, num_excitatory: int, num_inhibitory: int, w_max: float, w_min: float, w_inh: float,
                 is_cuda: bool, conductive: bool, p_mask: float = 0, name: str = 'pop'):
        """
        Initialize a Brader-style network with a single population of Izhikevich neurons.

        Args:
            num_excitatory: Number of excitatory neurons in the population
            num_inhibitory: Number of inhibitory neurons in the population
            w_max: Maximum weight for excitatory connections
            w_min: Minimum weight for excitatory connections
            w_inh: Weight for inhibitory connections
            is_cuda: Whether to use GPU acceleration
            conductive: Whether to use conductance-based synapses
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
            name: Name of the population in the network
        """
        super().__init__()
        pop_params = nu.SimpleExcitatoryParams(num_excitatory, is_cuda) + \
            nu.SimpleInhibitoryParams(num_inhibitory, is_cuda)
        pop = nu.IzhPopulation(pop_params, conductive)

        synaptic_cnxn = SimpleConnection(pop, pop, w_max, w_min, w_inh, True, p_mask=p_mask)
        self.add_population(pop, name)
        self.add_connection((name, name), synaptic_cnxn)
        self.name = name


class BoolNet(IzhNet):
    """
    A network for boolean logic tasks with input, hidden, and output layers.

    This class creates a network with multiple layers:
    - Input layer: Multiple input populations that can represent boolean values
    - Hidden layer: A single population of Izhikevich neurons
    - Output layer: Multiple output populations that can represent boolean values
    - Teacher layer: Populations that can provide supervised learning signals

    The network is designed for boolean logic tasks, where the inputs are boolean values
    and the network learns to produce the correct boolean output based on the inputs.
    The network uses Brader-style connections and can be tuned to achieve specific
    firing rates.
    """
    def __init__(self, n_inputs: int, n_e: int, n_i: int, w_max: float, w_min: float, w_inh: float, is_cuda: bool,
                 is_conductive: bool, p_mask: float = 0):
        """
        Initialize a boolean logic network.

        Args:
            n_inputs: Number of boolean inputs to the network
            n_e: Number of excitatory neurons in the hidden layer
            n_i: Number of inhibitory neurons in the hidden layer
            w_max: Maximum weight for excitatory connections
            w_min: Minimum weight for excitatory connections
            w_inh: Weight for inhibitory connections
            is_cuda: Whether to use GPU acceleration
            is_conductive: Whether to use conductance-based synapses
            p_mask: Probability of masking a connection (0 = no masking, 1 = all masked)
        """
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
        """
        Run the network with boolean inputs and a target output.

        This method converts boolean inputs and target into firing rates for the
        input and teacher populations, and then runs a step of the network.

        For each boolean input:
        - If the input is 0, the corresponding input_0 population fires at the quiescent rate
          and the input_1 population fires at the active rate
        - If the input is 1, the corresponding input_0 population fires at the active rate
          and the input_1 population fires at the quiescent rate

        For the target output:
        - If the target is 0, the teacher_0 population fires at the quiescent rate
          and the teacher_1 population fires at the active rate
        - If the target is 1, the teacher_0 population fires at the active rate
          and the teacher_1 population fires at the quiescent rate

        Args:
            boolean_input: Array of boolean inputs (0 or 1)
            target: Target output (0 or 1)
        """
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
        Adjust ionput rate so that the average firing ratio across
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
        """
        Tune the quiescent firing rate to achieve a target firing rate.

        This method adjusts the quiescent_rate parameter so that the average firing ratio
        across the specified populations approaches the target rate. It uses the tune_to_rate
        method to perform the tuning.

        Args:
            target_rate: Desired firing ratio, e.g. 0.15 for 15% firing
            populations_to_monitor: Set of population names whose firing rates we average
            steps_checked: Number of simulation steps per tuning iteration
            max_iters: Maximum number of iterations to attempt
            tolerance: Stop when |avg_rate - target_rate| < tolerance

        Returns:
            The tuned quiescent rate
        """
        self.quiescent_rate = self.tune_to_rate(target_rate, populations_to_monitor, steps_checked, max_iters, tolerance)

    def tune_active_rate(self, target_rate: float, populations_to_monitor: set[str],
                         steps_checked: int = 5000,  # Check on full second
                         max_iters: int = 20, tolerance: float = 1e-3):
        """
        Tune the active firing rate to achieve a target firing rate.

        This method adjusts the active_rate parameter so that the average firing ratio
        across the specified populations approaches the target rate. It uses the tune_to_rate
        method to perform the tuning.

        Args:
            target_rate: Desired firing ratio, e.g. 0.15 for 15% firing
            populations_to_monitor: Set of population names whose firing rates we average
            steps_checked: Number of simulation steps per tuning iteration
            max_iters: Maximum number of iterations to attempt
            tolerance: Stop when |avg_rate - target_rate| < tolerance

        Returns:
            The tuned active rate
        """
        self.active_rate = self.tune_to_rate(target_rate, populations_to_monitor, steps_checked, max_iters, tolerance)

    def read_outputs(self):
        """
        Read the firing rates from both output populations and return the softmax of the rates.

        This method calculates the firing rates of the two output populations ('output_pop_0'
        and 'output_pop_1') and applies the softmax function to convert them into probabilities
        that sum to 1. The softmax function is defined as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

        Returns:
            An array of probabilities [probability_of_0, probability_of_1]
        """
        if self.device == 'cuda':
            dev = cp
        else:
            dev = np

        # Get the firing rates from both output populations
        rates = []
        for pop_name in self.output_pops:
            rates.append(self.firing_populations[pop_name].firing_ratio())

        # Convert to array for softmax calculation
        rates = dev.array(rates)

        # Apply softmax function
        exp_rates = dev.exp(rates)
        softmax_rates = exp_rates / dev.sum(exp_rates)

        return softmax_rates





def brader_init(connection: SynapticConnection, w_high: float, w_low: float,
                 w_inhibitory: float, p_connected_e: float, p_connected_i: float):
    """
    Initialize a connection with Brader-style weights.

    This function initializes a synaptic connection with Brader-style weights:
    - Excitatory connections are set to either w_high or w_low
    - Inhibitory connections are set to w_inhibitory
    - Connections are randomly masked based on p_connected_e and p_connected_i

    This initialization is based on the model described in Brader et al. (2007),
    which is used for spike-timing-dependent plasticity (STDP) learning.

    Args:
        connection: The synaptic connection to initialize
        w_high: High weight value for excitatory connections
        w_low: Low weight value for excitatory connections
        w_inhibitory: Weight value for inhibitory connections
        p_connected_e: Probability of connecting excitatory neurons
        p_connected_i: Probability of connecting inhibitory neurons
    """
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
    """
    Initialize a connection with all zero weights.

    This function sets all weights in a synaptic connection to zero.
    This is useful for creating connections that will be learned from scratch.

    Args:
        connection: The synaptic connection to initialize
    """
    connection.connection[:, :] = 0.0
