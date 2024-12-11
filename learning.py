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
        self.d_connections += (                                                               # piecewise function
            (post_minus_pre < 0) * self.a_minus * dev.exp(post_minus_pre / self.tau_minus) +  # Negative branch
            (post_minus_pre > 0) * self.a_plus * dev.exp(-post_minus_pre / self.tau_plus)     # Positive branch
        ) * dev.logical_or.outer(self.network.fired, self.network.fired)                      # only update when fired

        self.network.connections += self.d_connections * self.network.mask
        self.network.connections[self.network.inhibitory] = (
            self.network.connections[self.network.inhibitory].clip(self.min_i, self.max_i))
        self.network.connections[self.network.inhibitory == False] = (
            self.network.connections[self.network.inhibitory == False].clip(self.min_e, self.max_e))




