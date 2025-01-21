import cupy as cp
import numpy as np
from networks import IzhNet, IzhParams


class LearningMechanism:
    def __init__(self, network: IzhNet):
        self.network = network
        
    def update_weights(self):
        raise NotImplemented
    

class SimpleSTDP(LearningMechanism):
    def __init__(self, network: IzhNet, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float,
                 min_e: float = 0, max_e: float = .5, min_i: float = -.5, max_i: float = 0,
                 alpha: float = 10e-6, beta: float = 10e-4):
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
        self.t_fired = dev.full_like(self.network.v, 0)
        self.d_connections = dev.full_like(self.network.connections, 0)

    def update_weights(self):
        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        # quiescent dynamics
        self.d_connections = self.d_connections - (self.d_connections - self.alpha) * self.beta

        # time since neurons fired
        self.t_fired[self.network.fired] = 0
        post_minus_pre = np.subtract.outer(self.t_fired, self.t_fired)

        # if dev.any(pre_minus_post < 0):
        #     breakpoint()

        self.t_fired += 1
        # breakpoint()
        self.d_connections += (                                                                # piecewise function
            (post_minus_pre < 0) * self.a_plus * dev.exp(post_minus_pre / self.tau_plus) +     # Negative branch
            (post_minus_pre > 0) * self.a_minus * dev.exp(-post_minus_pre / self.tau_minus)       # Positive branch
        ) * dev.logical_or.outer(self.network.fired, self.network.fired)                      # only update when fired

        # if any(self.network.fired):
        #     breakpoint()

        self.network.connections += self.d_connections * self.network.mask
        self.network.connections[self.network.inhibitory] = (
            self.network.connections[self.network.inhibitory].clip(self.min_i, self.max_i))
        self.network.connections[self.network.inhibitory == False] = (
            self.network.connections[self.network.inhibitory == False].clip(self.min_e, self.max_e))


class BraderSTDP(LearningMechanism):
    def __init__(self, network: IzhNet, x_max: float, voltage_threshold: float, theta_up_l: float, theta_up_h: float,
                 theta_down_l: float, theta_down_h: float, theta_x: float, up_increment: float, down_increment: float,
                 c_increment: float, tau_c: float, alpha: float, beta: float, j_negative: float, j_positive: float,
                 j_inhibitory: float):
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
        self.x = dev.full_like(self.network.connections, self.theta_x)

        # quiescent parameters
        self.alpha = alpha
        self.beta = beta

        # calcium dynamics
        self.tau_c = tau_c
        self.c_inc = c_increment
        self.c = dev.zeros_like(self.network.fired)

    def update_weights(self):
        if self.network.device == 'cuda':
            dev = cp
        else:
            dev = np

        # calcium update
        self.c = self.c - self.c/self.tau_c
        self.c[self.network.fired] += self.c_inc

        # calculate which post-synaptic neurons have their synapses eligible for updates
        x_increase_post = dev.logical_and(self.network.v > self.voltage_threshold, self.theta_u_l < self.c,
                                          self.c < self.theta_u_h)

        x_decrease_post = dev.logical_and(self.network.v <= self.voltage_threshold, self.theta_d_l < self.c,
                                          self.c < self.theta_d_h)

        # calculate which synapses are eligible for updates
        # (when the post-synaptic neuron is eligible and the pre-synaptic neuron fires)
        x_increase_synapse = dev.logical_and.outer(x_increase_post, self.network.fired)

        x_decrease_synapse = dev.logical_and.outer(x_decrease_post, self.network.fired)

        # which synapses are not updated
        x_quiescent = dev.logical_not(dev.logical_or(x_increase_synapse, x_decrease_synapse))

        self.x[x_increase_synapse] += self.u_inc
        self.x[x_decrease_synapse] -= self.d_inc
        self.x[x_quiescent] += (self.alpha*(self.x[x_quiescent] > self.theta_x) +  # above threshold branch
                                self.beta*(self.x[x_quiescent] <= self.theta_x))   # below threshold branch

        # finally, synaptic weights are set to one of four values
        pos_weights = self.x > self.theta_x
        self.network.connections[pos_weights] = self.j_pos
        self.network.connections[dev.logical_not(pos_weights)] = self.j_neg
        self.network.connections[:, self.network.inhibitory] = self.j_inh
        self.network.connections[self.network.mask] = 0




