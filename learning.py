"""
This module provides classes for implementing learning mechanisms in Izhikevich neural networks.

The module defines different learning mechanisms that can be used to update the weights
of connections between neuron populations in an Izhikevich network. These mechanisms
implement different forms of synaptic plasticity, which is the ability of synapses to
strengthen or weaken over time.

The main classes are:
1. LearningMechanism: Base class for all learning mechanisms
2. SimpleSTDP: Implementation of spike-timing-dependent plasticity (STDP)
3. BraderSTDP: Implementation of the Brader method of synaptic plasticity, which
   assumes that all excitatory synapses take one of two values (binary weights)
"""

import cupy as cp
import numpy as np
import networks as nt
import neurons as nu


class LearningMechanism:
    """
    Base class for all learning mechanisms.

    This abstract class defines the interface for learning mechanisms that can be used
    to update the weights of connections between neuron populations in an Izhikevich network.
    Subclasses implement specific learning rules, such as spike-timing-dependent plasticity
    (STDP) or the Brader method.

    Learning mechanisms are used by the IzhNetTrainer class in the training module to
    update the weights of connections during training.
    """
    def __init__(self, network: nt.IzhNet):
        """
        Initialize the learning mechanism with a network.

        Args:
            network: The IzhNet instance whose weights will be updated by this learning mechanism.
        """
        self.network = network

    def update_weights(self):
        """
        Update the weights of connections in the network.

        This method is called during training to update the weights of connections
        based on the activity of the network. It must be implemented by subclasses.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplemented


class SimpleSTDP(LearningMechanism):
    """
    Implementation of spike-timing-dependent plasticity (STDP).

    This class implements a standard form of spike-timing-dependent plasticity (STDP),
    which is a biological process that adjusts the strength of connections between neurons
    based on the relative timing of a particular neuron's output and input spikes.

    The STDP rule implemented here has the following characteristics:
    - If a postsynaptic neuron fires shortly after a presynaptic neuron, the connection
      is strengthened (long-term potentiation, LTP).
    - If a postsynaptic neuron fires shortly before a presynaptic neuron, the connection
      is weakened (long-term depression, LTD).
    - The amount of strengthening or weakening depends on the time difference between
      the spikes, with a larger effect for smaller time differences.

    The weights are constrained to be within specified ranges for excitatory and inhibitory
    connections, and there is also a quiescent dynamics that slowly pulls the weights
    towards a baseline value when there is no activity.
    """
    def __init__(self, network: nt.IzhNet, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float,
                 min_e: float = 0, max_e: float = .5, min_i: float = -.5, max_i: float = 0,
                 alpha: float = 10e-6, beta: float = 10e-4):
        """
        Initialize the STDP learning mechanism.

        Args:
            network: The IzhNet instance whose weights will be updated by this learning mechanism.
            a_plus: The amount of strengthening (LTP) for each spike pair.
            a_minus: The amount of weakening (LTD) for each spike pair.
            tau_plus: The time constant for the strengthening (LTP) effect.
            tau_minus: The time constant for the weakening (LTD) effect.
            min_e: The minimum weight for excitatory connections.
            max_e: The maximum weight for excitatory connections.
            min_i: The minimum weight for inhibitory connections.
            max_i: The maximum weight for inhibitory connections.
            alpha: The baseline value for quiescent dynamics.
            beta: The rate of return to baseline for quiescent dynamics.
        """
        super(SimpleSTDP, self).__init__(network)
        self.alpha = alpha
        self.beta = beta
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.min_i = min_i
        self.max_i = max_i
        self.min_e = min_e
        self.max_e = max_e

        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        self.device = 'cuda'
        self.t_fireds = {}
        self.d_connections = {}
        for (pre_pop, post_pop), connection in self.network.population_connections.items():
            if isinstance(post_pop, nu.IzhPopulation):
                self.t_fired = dev.full_like(self.network.firing_populations[pre_pop].fired, 0)
                self.t_fired = dev.full_like(self.network.firing_populations[post_pop].fired, 0)
                self.d_connections[(pre_pop, post_pop)] = dev.full_like(connection, 0)

    def update_weights(self):
        """
        Update the weights of connections in the network using STDP.

        This method implements the STDP learning rule, which updates the weights of connections
        based on the relative timing of pre- and post-synaptic spikes. The method performs
        the following steps:

        1. For each population, update the time since neurons fired.
        2. For each connection, if it can be updated:
           a. Apply quiescent dynamics to slowly pull weights towards a baseline value.
           b. Calculate the time difference between post- and pre-synaptic spikes.
           c. Update the connection weights based on the STDP rule.
           d. Clip the weights to be within the specified ranges for excitatory and inhibitory connections.

        The STDP rule strengthens connections where the post-synaptic neuron fires shortly after
        the pre-synaptic neuron, and weakens connections where the post-synaptic neuron fires
        shortly before the pre-synaptic neuron.
        """
        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        for pop, t_fired in self.t_fireds:
            # update time since neurons fired
            self.t_fireds[pop][self.network.firing_populations[pop].fired] = 0

        for pops, d_connection in self.d_connections.items():
            if self.network.population_connections[pops].can_update:
                pre_pop, post_pop = pops
                pre_fired = self.network.firing_populations[pre_pop].fired
                post_fired = self.network.firing_populations[post_pop].fired

                # quiescent dynamics
                d_connection = d_connection - (d_connection - self.alpha) * self.beta

                # calculate difference between post and pre
                post_minus_pre = dev.subtract.outer(self.t_fireds[post_fired], self.t_fireds[pre_fired]).transpose()

                # update t_fired after using it
                for pop in self.t_fireds.keys():
                    self.t_fireds[pop] += 1

                # update d_connection (locally)
                d_connection += (                                                                # piecewise function
                    (post_minus_pre < 0) * self.a_plus * dev.exp(post_minus_pre / self.tau_plus) +     # Negative branch
                    (post_minus_pre > 0) * self.a_minus * dev.exp(-post_minus_pre / self.tau_minus)       # Positive branch
                ) * dev.logical_or.outer(pre_fired, post_fired)                  # only update when fired

                # pull inhibitory
                inhib = self.network.firing_populations[pre_pop].inhibitory

                self.network.population_connections[pops] += d_connection * self.network.population_connections[pops].mask
                self.network.population_connections[pops][inhib] = (
                    self.network.population_connections[pops][inhib].clip(self.min_i, self.max_i))
                self.network.population_connections[pops][inhib==False] = (
                    self.network.population_connections[pops][inhib==False].clip(self.min_e, self.max_e))

                # update d_connection for pop
                self.d_connections[pops] = d_connection


class BraderSTDP(LearningMechanism):
    """
    Implementation of the Brader method of synaptic plasticity.

    This class implements the Brader method of synaptic plasticity, which is a form of
    spike-timing-dependent plasticity (STDP) that assumes all excitatory synapses take
    one of two values (binary weights). This method is based on the model described in
    Brader et al. (2007).

    The Brader method uses a calcium-based model of synaptic plasticity, where the
    eligibility of a synapse for potentiation or depression depends on the calcium
    concentration in the post-synaptic neuron and its membrane potential. The method
    has the following characteristics:

    - Synapses are potentiated when the post-synaptic membrane potential is high and
      the calcium concentration is within a specific range.
    - Synapses are depressed when the post-synaptic membrane potential is low and
      the calcium concentration is within a specific range.
    - The calcium concentration increases when the post-synaptic neuron fires and
      decays exponentially over time.
    - Synaptic weights are binary, taking one of two values (j_positive or j_negative)
      for excitatory synapses, and a fixed value (j_inhibitory) for inhibitory synapses.

    This method is particularly useful for learning tasks where binary decision boundaries
    are appropriate, such as classification tasks.
    """
    def __init__(self, network: nt.IzhNet, x_max: float, voltage_threshold: float, theta_up_l: float, theta_up_h: float,
                 theta_down_l: float, theta_down_h: float, theta_x: float, up_increment: float, down_increment: float,
                 c_increment: float, tau_c: float, alpha: float, beta: float, j_negative: float, j_positive: float,
                 j_inhibitory: float):
        """
        Initialize the Brader STDP learning mechanism.

        Args:
            network: The IzhNet instance whose weights will be updated by this learning mechanism.
            x_max: The maximum value for the eligibility trace.
            voltage_threshold: The threshold voltage for determining whether a neuron is in a high or low state.
            theta_up_l: The lower bound of the calcium concentration range for potentiation.
            theta_up_h: The upper bound of the calcium concentration range for potentiation.
            theta_down_l: The lower bound of the calcium concentration range for depression.
            theta_down_h: The upper bound of the calcium concentration range for depression.
            theta_x: The threshold for the eligibility trace to determine the synaptic weight.
            up_increment: The amount to increment the eligibility trace for potentiation.
            down_increment: The amount to decrement the eligibility trace for depression.
            c_increment: The amount to increment the calcium concentration when a neuron fires.
            tau_c: The time constant for the decay of the calcium concentration.
            alpha: The rate of increase for the eligibility trace during quiescent periods when above threshold.
            beta: The rate of increase for the eligibility trace during quiescent periods when below threshold.
            j_negative: The weight value for excitatory synapses when the eligibility trace is below threshold.
            j_positive: The weight value for excitatory synapses when the eligibility trace is above threshold.
            j_inhibitory: The weight value for inhibitory synapses.
        """
        super(BraderSTDP, self).__init__(network)

        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        # weight update parameters
        self.voltage_threshold = voltage_threshold
        self.theta_d_l = theta_down_l
        self.theta_d_h = theta_down_h
        self.theta_u_l = theta_up_l
        self.theta_u_h = theta_up_h
        self.u_inc = up_increment
        self.d_inc = down_increment
        self.j_pos = j_positive
        self.j_neg = j_negative
        self.j_inh = j_inhibitory
        self.theta_x = theta_x
        self.x_max = x_max
        self.connection_xs = {}
        self.population_cs = {}

        for (pop_1, pop_2), connection in self.network.population_connections:
            if isinstance(pop_2, nu.IzhPopulation):
                self.connection_xs[(pop_1, pop_2)] = dev.full_like(connection, self.theta_x)
                neurons = self.network.firing_populations[pop_1]
                self.population_cs[pop_2] = dev.zeros_like(neurons)

        # quiescent parameters
        self.alpha = alpha
        self.beta = beta

        # calcium dynamics
        self.tau_c = tau_c
        self.c_inc = c_increment

        self.population_cs = {}

    def update_weights(self, update_inibitory: bool = False):
        """
        Update the weights of connections in the network using the Brader method.

        This method implements the Brader method of synaptic plasticity, which updates
        the weights of connections based on the calcium concentration in the post-synaptic
        neuron and its membrane potential. The method performs the following steps:

        1. For each connection, if it can be updated:
           a. Get the post-synaptic neuron's membrane potential and calcium concentration.
           b. Update the calcium concentration based on whether the neuron fired.
           c. Determine which synapses are eligible for potentiation or depression based
              on the calcium concentration and membrane potential.
           d. Update the eligibility trace for each synapse.
           e. Set the synaptic weights based on the eligibility trace, with excitatory
              synapses taking one of two values (j_positive or j_negative) and inhibitory
              synapses taking a fixed value (j_inhibitory).

        Args:
            update_inibitory: Whether to update the weights of inhibitory synapses.
                             If True, inhibitory synapses are set to j_inhibitory.
                             If False, inhibitory synapses are not updated.
        """
        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        for pops, x in self.connection_xs:
            if self.network.population_connections[pops].can_update:
                pre_pop, post_pop = pops
                pre_fired = self.network.firing_populations[pre_pop].fired
                post_fired = self.network.firing_populations[post_pop].fired

                # get post-synaptic neuron info
                v = self.network.firing_populations[post_pop].v
                c = self.population_cs[post_pop]

                # calcium update
                c = c - c/self.tau_c
                c[post_fired] += self.c_inc

                # calculate which post-synaptic neurons have their synapses eligible for updates
                x_increase_post = dev.logical_and(v > self.voltage_threshold, self.theta_u_l < c,
                                                  c < self.theta_u_h)

                x_decrease_post = dev.logical_and(v <= self.voltage_threshold, self.theta_d_l < c,
                                                  c < self.theta_d_h)

                # calculate which synapses are eligible for updates
                # (when the post-synaptic neuron is eligible and the pre-synaptic neuron fires)
                x_increase_synapse = dev.logical_and.outer(x_increase_post, pre_fired)

                x_decrease_synapse = dev.logical_and.outer(x_decrease_post, pre_fired)

                # which synapses are not updated
                x_quiescent = dev.logical_not(dev.logical_or(x_increase_synapse, x_decrease_synapse))

                self.connection_xs[pops][x_increase_synapse] += self.u_inc
                self.connection_xs[pops][x_decrease_synapse] -= self.d_inc
                self.connection_xs[pops][x_quiescent] += (self.alpha*(self.connection_xs[x_quiescent] > self.theta_x) +  # above threshold branch
                                        self.beta*(self.connection_xs[x_quiescent] <= self.theta_x))   # below threshold branch

                # finally, synaptic weights are set to one of four values
                pos_weights = self.connection_xs[pops] > self.theta_x
                self.network.population_connections[pops][pos_weights] = self.j_pos
                self.network.population_connections[pops][dev.logical_not(pos_weights)] = self.j_neg
                if update_inibitory:
                    self.network.population_connections[pops][:, self.network.firing_populations[pre_pop].inhibitory] = self.j_inh
                self.network.population_connections[pops][self.network.population_connections[pops].mask] = 0

                # update c variable
                self.population_cs[post_pop] = c
