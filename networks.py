import numpy as np
import cupy as cp
from typing import Union, Optional

GenArray = Union[cp.ndarray, np.ndarray]  # used repeatedly throughout code

# From Izhikevich et al. 2004
DEFAULT_TAU_A = 5
DEFAULT_TAU_B = 150
DEFAULT_TAU_C = 6
DEFAULT_TAU_D = 150


class IzhParams:
    """
    Parameters for Izhikevich networks
    TODO: build functions which return parameterizations
    """
    def __init__(self, a: GenArray, b: GenArray, c: GenArray, d: GenArray, inhibitory: GenArray,
                 U: GenArray, F: GenArray, D: GenArray):
        assert a.shape == b.shape == c.shape == d.shape == U.shape == F.shape == D.shape == inhibitory.shape
        assert type(a) == type(b) == type(c) == type(d) == type(U) == type(F) == type(D) == type(inhibitory)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inhibitory = inhibitory
        self.U = U
        self.F = F
        self.D = D

    def __add__(self, other: "IzhParams"):
        """
        :param other: Another IzhParams object
        :return: concatenation of all IzhParam parameters as new IzhParams object
        """
        assert type(self.a) == type(other.a) and isinstance(self.a, GenArray)
        if isinstance(self.a, cp.ndarray):
            dev = cp
        else:
            dev = np
        new_a = dev.concatenate([self.a, other.a])
        new_b = dev.concatenate([self.b, other.b])
        new_c = dev.concatenate([self.c, other.c])
        new_d = dev.concatenate([self.d, other.d])
        new_inhibitory = dev.concatenate([self.inhibitory, other.inhibitory])
        new_U = dev.concatenate([self.U, other.U])
        new_F = dev.concatenate([self.F, other.F])
        new_D = dev.concatenate([self.D, other.D])

        return IzhParams(new_a, new_b, new_c, new_d, new_inhibitory, new_U, new_F, new_D)


# Cortical excitatory, constant paramterizations
class RSParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .02)
        b = dev.full(length, .2)
        c = dev.full(length, -65.0)
        d = dev.full(length, 8.0)
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


class IBParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .02)
        b = dev.full(length, .2)
        c = dev.full(length, -55.0)
        d = dev.full(length, 4.0)
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)

class CHParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .02)
        b = dev.full(length, .2)
        c = dev.full(length, -50.0)
        d = dev.full(length, 2.0)
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


# Cortical inhibitory, constant paramterizations
class FSParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .1)
        b = dev.full(length, .2)
        c = dev.full(length, -65.0)
        d = dev.full(length, 2.0)
        inhibitory = dev.full(length, True)
        U = dev.full(length, 0.2)
        F = dev.full(length, 20.0)
        D = dev.full(length, 700.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


class LTSParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .02)
        b = dev.full(length, .25)
        c = dev.full(length, -65.0)
        d = dev.full(length, 2.0)
        inhibitory = dev.full(length, True)
        U = dev.full(length, 0.2)
        F = dev.full(length, 20.0)
        D = dev.full(length, 700.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


# "Other" neurons
class RZParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .1)
        b = dev.full(length, .25)
        c = dev.full(length, -65.0)
        d = dev.full(length, 2.0)
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


class TCParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        a = dev.full(length, .25)
        b = dev.full(length, .02)
        c = dev.full(length, -65.0)
        d = dev.full(length, .05)
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


# Distribution corresponding to original paper

class SimpleExcitatoryParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        r = dev.random.rand(length)
        a = dev.full(length, .02)
        b = dev.full(length, .2)
        c = 15*r**2 - 65
        d = 8-6*r**2
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


class SimpleInhibitoryParams(IzhParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        r = dev.random.rand(length)
        a = .02+0.08*r
        b = 0.25-0.05*r
        c = dev.full(length, -65.0)
        d = dev.full(length, 2.0)
        inhibitory = dev.full(length, True)
        U = dev.full(length, 0.2)
        F = dev.full(length, 20.0)
        D = dev.full(length, 700.0)
        super().__init__(a, b, c, d, inhibitory, U, F, D)


class IzhNet:
    """
    An Izhikevich network
    """
    def __init__(self, params: IzhParams, connections: GenArray, conductive: bool, mask: Optional[GenArray] = None,
                 tau_a: float = DEFAULT_TAU_A, tau_b: float = DEFAULT_TAU_B,
                 tau_c: float = DEFAULT_TAU_C, tau_d: float = DEFAULT_TAU_D):
        if type(params.a) == cp.ndarray:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        assert type(params.a) == type(connections)

        # standard params
        self.a = params.a
        self.b = params.b
        self.c = params.c
        self.d = params.d

        self.inhibitory = params.inhibitory

        # conductance params
        self.U = params.U
        self.F = params.F
        self.D = params.D

        self.tau_a = tau_a
        self.tau_b = tau_b
        self.tau_c = tau_c
        self.tau_d = tau_d
        self.connections = connections
        self.conductive = conductive

        if self.device == 'cuda':
            dev = cp
        else:
            dev = np

        if mask is None:
            self.mask = dev.ones_like(self.connections)
        else:
            self.mask = mask

        self.connections *= self.connections * self.mask

        self.fired = dev.zeros_like(self.a).astype(bool)
        self.v = dev.full_like(self.a, -65.0)  # starting voltage
        self.u = self.b * self.v

        # conductance variables
        self.g_a = dev.zeros_like(self.a)
        self.g_b = dev.zeros_like(self.a)
        self.g_c = dev.zeros_like(self.a)
        self.g_d = dev.zeros_like(self.a)
        self.R = dev.zeros_like(self.a)
        self.w = dev.zeros_like(self.a)

    def step(self, input_voltages: GenArray):
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
            self.conductance_update()
            self.facilitation_update()

    def __call__(self, input_voltages: np.ndarray):
        self.step(input_voltages)

    def is_cuda(self, is_cuda: bool):
        """
        Switches device for network
        :param is_cuda: whether network is on cuda device
        :return: None
        """

        if is_cuda:
            self.device = 'cuda'
            self.a = cp.array(self.a)
            self.b = cp.array(self.b)
            self.c = cp.array(self.c)
            self.d = cp.array(self.d)
            self.fired = cp.array(self.fired)
            self.v = cp.array(self.v)
            self.u = cp.array(self.u)
            self.U = cp.array(self.U)
            self.F = cp.array(self.F)
            self.D = cp.array(self.D)
            self.connections = cp.array(self.connections)
        else:
            self.device = 'cpu'
            self.a = cp.asnumpy(self.a)
            self.b = cp.asnumpy(self.b)
            self.c = cp.asnumpy(self.c)
            self.d = cp.asnumpy(self.d)
            self.fired = cp.asnumpy(self.fired)
            self.v = cp.asnumpy(self.v)
            self.u = cp.asnumpy(self.u)
            self.U = cp.asnumpy(self.U)
            self.F = cp.asnumpy(self.F)
            self.D = cp.asnumpy(self.D)
            self.connections = cp.asnumpy(self.connections)

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

    def conductance_update(self):
        """
        :return:
        """
        self.g_a = self.g_a + self.connections @ (
                self.R * self.w * self.fired * (self.inhibitory == 0)) - self.g_a / self.tau_a
        self.g_b = self.g_b + self.connections @ (
                self.R * self.w * self.fired * (self.inhibitory == 0)) - self.g_b / self.tau_b
        self.g_c = self.g_c + self.connections @ (
                self.R * self.w * self.fired * self.inhibitory) - self.g_c / self.tau_c
        self.g_d = self.g_d + self.connections @ (
                self.R * self.w * self.fired * self.inhibitory) - self.g_d / self.tau_d

    def facilitation_update(self):
        """
        :return:
        """
        self.R = self.R + (1-self.R)/self.D - self.R * self.w * self.fired
        self.w = self.w + (self.U - self.w)/self.F + self.U * (1-self.w) * self.fired


class SimpleNetwork(IzhNet):
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