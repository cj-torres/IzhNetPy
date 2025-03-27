"""
This module provides classes for creating and simulating Izhikevich neuron populations.

The module is organized into two main types of classes:
1. Parameter classes (IzhParams, InputParams and their subclasses) - These define parameters
   for different types of neuron populations. They are used to make initialization of common
   population types easier.
2. Population classes (NeuronPopulation and subclasses) - These implement the actual neuron
   populations that can be used in networks.

There are two main types of populations:
- Regular populations (IzhPopulation) - These use full Izhikevich parameters and implement
  the complete Izhikevich neuron model.
- Input populations (InputPopulation) - These are simpler and don't need full Izhikevich
  parameters, but still take parameters which model facilitatory variables.
"""

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
    Parameters for Izhikevich neuron populations.

    This class defines the parameters needed for the Izhikevich neuron model, which is a simplified
    model that can reproduce various firing patterns observed in real neurons. The parameters are:

    - a: Time scale of the recovery variable u
    - b: Sensitivity of the recovery variable u to the membrane potential v
    - c: After-spike reset value of the membrane potential v
    - d: After-spike reset increment of the recovery variable u
    - inhibitory: Boolean indicating whether the neuron is inhibitory
    - U, F, D: Parameters for the short-term plasticity model (facilitation and depression)

    This class is used as input to IzhPopulation to create populations of neurons with specific
    firing characteristics. Subclasses of IzhParams define specific neuron types with preset
    parameter values.
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


# Cortical inhibitory, constant parameterizations
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


class InputParams:
    """
    Parameters for input neuron populations.

    This class defines a simplified set of parameters for input populations that don't need
    the full Izhikevich neuron model. Input populations are used to provide external input
    to a network, often with stochastic firing patterns (like Poisson processes).

    The parameters are:
    - inhibitory: Boolean indicating whether the input is inhibitory
    - U, F, D: Parameters for the short-term plasticity model (facilitation and depression)

    This class is used as input to InputPopulation to create populations of input neurons.
    Subclasses define specific input types with preset parameter values.
    """
    def __init__(self,  inhibitory: GenArray, U: GenArray, F: GenArray, D: GenArray):
        self.U = U
        self.F = F
        self.D = D
        self.inhibitory = inhibitory

    def __add__(self, other: "InputParams"):
        """
        :param other: Another IzhParams object
        :return: concatenation of all IzhParam parameters as new IzhParams object
        """
        assert type(self.U) == type(other.U) and isinstance(self.U, GenArray)
        if isinstance(self.U, cp.ndarray):
            dev = cp
        else:
            dev = np
        new_inhibitory = dev.concatenate([self.inhibitory, other.inhibitory])
        new_U = dev.concatenate([self.U, other.U])
        new_F = dev.concatenate([self.F, other.F])
        new_D = dev.concatenate([self.D, other.D])

        return InputParams(new_inhibitory, new_U, new_F, new_D)


class SimpleExcitatoryInputParams(InputParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        inhibitory = dev.full(length, False)
        U = dev.full(length, .5)
        F = dev.full(length, 1000.0)
        D = dev.full(length, 800.0)
        super().__init__(inhibitory, U, F, D)


class SimpleInhibitoryInputParams(InputParams):
    def __init__(self, length: int, is_cuda: bool):
        if is_cuda:
            dev = cp
        else:
            dev = np
        inhibitory = dev.full(length, True)
        U = dev.full(length, 0.2)
        F = dev.full(length, 20.0)
        D = dev.full(length, 700.0)
        super().__init__(inhibitory, U, F, D)


class NeuronPopulation:
    """
    Base class for all neuron populations.

    This abstract class defines the common interface and functionality for all types of neuron
    populations in the network. Subclasses implement specific types of populations, such as:
    - IzhPopulation: Full Izhikevich neuron populations
    - InputPopulation: Simplified input populations
    - GaussianPopulation: Populations that output Gaussian noise

    All populations can be run on either CPU (using NumPy) or GPU (using CuPy), and they
    can be either regular populations or input populations.
    """
    def __init__(self, is_cuda: bool, is_input: bool):
        if is_cuda:
            self.device = 'cuda'
            dev = cp
        else:
            self.device = 'cpu'
            dev = np
        self.fired = dev.ndarray((0,))
        self.inhibitory = dev.ndarray((0,))
        self.parameters = ['fired']
        self.is_input = is_input

    def is_cuda(self, is_cuda: bool):
        """
        Switches device for network
        :param is_cuda: whether network is on cuda device
        :return: None
        """
        self.device = 'cuda' if is_cuda else 'cpu'
        for param_name in self.parameters:
            arr = getattr(self, param_name)  # get the attribute
            if self.device == 'cuda':
                # move to GPU (cupy)
                if not isinstance(arr, cp.ndarray):
                    arr = cp.asarray(arr)
            else:
                # move to CPU (numpy)
                if not isinstance(arr, np.ndarray):
                    arr = cp.asnumpy(arr)
            setattr(self, param_name, arr)

    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)

    def __len__(self):
        return len(self.fired)

    def step(self, *args, **kwargs):
        raise NotImplemented

    def get_output(self):
        raise NotImplemented

    def firing_ratio(self):
        return self.fired.mean()

    def reset(self):
        pass


class GaussianPopulation(NeuronPopulation):
    """
    A population that outputs Gaussian noise.

    This is a simple input population that generates output values from a Gaussian (normal)
    distribution with specified mean and standard deviation. It's useful for providing
    random input to a network with a specific distribution.

    Unlike other populations, this one doesn't implement the Izhikevich model or use
    facilitatory variables. It simply outputs random values on each step.
    """
    def __init__(self, mean: GenArray, std: GenArray, is_cuda: bool):
        if is_cuda:
            self.device = 'cuda'
            dev = cp
        else:
            self.device = 'cpu'
            dev = np
        super().__init__(is_cuda, True)
        self.mean = mean
        self.std = std
        self.fired = dev.full_like(self.mean, True)
        self.inhibitory = dev.full_like(self.mean, False)
        self.parameters.extend(['mean', 'std'])

    def get_output(self):
        if self.device == 'cuda':
            dev = cp
        else:
            dev = np
        return dev.random.normal(loc=self.mean, scale=self.std)

    def step(self):
        pass


class InputPopulation(NeuronPopulation):
    """
    A population that provides input to a network.

    This class implements a simplified neuron population that doesn't use the full Izhikevich
    model, but still models facilitatory variables (U, F, D) for short-term plasticity. It's
    used to provide external input to a network, typically with stochastic firing patterns.

    Unlike IzhPopulation, this class doesn't have membrane potential dynamics. Instead, neurons
    fire randomly based on a specified firing rate. However, the output can still be modulated
    by facilitatory variables if the population is conductive.

    This class takes InputParams as input to initialize the population parameters.
    """
    def __init__(self, params: InputParams, conductive: bool):
        if type(params.U) == cp.ndarray:
            dev = cp
            is_cuda = True
        else:
            is_cuda = False
            dev = np

        super().__init__(is_cuda, True)

        self.device = "cuda" if is_cuda else "cpu"
        self.conductive = conductive

        self.inhibitory = params.inhibitory

        self.parameters.append('inhibitory')

        self.U = params.U
        self.parameters.append('U')
        self.F = params.F
        self.parameters.append('F')
        self.D = params.D
        self.parameters.append('D')
        self.R = dev.zeros_like(self.U)
        self.parameters.append('R')
        self.w = dev.zeros_like(self.U)
        self.parameters.append('w')
        self.fired = dev.zeros_like(self.U)
        self.parameters.append('fired')

    def facilitation_update(self):
        """
        Update the facilitation variables for short-term synaptic plasticity in input neurons.

        This method implements the dynamics of short-term synaptic plasticity for input
        populations, which includes both facilitation and depression. The model is based on
        the Tsodyks-Markram model, where:
        - R represents the fraction of available synaptic resources (depression)
        - w represents the synaptic efficacy (facilitation)

        The variables are updated based on the firing state of the input neurons and
        the parameters U (baseline release probability), F (facilitation time constant),
        and D (depression time constant).
        """
        self.R = self.R + (1-self.R)/self.D - self.R * self.w * self.fired
        self.w = self.w + (self.U - self.w)/self.F + self.U * (1-self.w) * self.fired

    def get_output(self):
        if self.conductive:
            return self.fired * self.w * self.R
        else:
            return self.fired

    def step(self, fire_rate: float):
        if self.device == "cuda":
            dev = cp
        else:
            dev = np
        self.fired = dev.random.rand(self.inhibitory.size) < fire_rate
        # self.facilitation_update()
        return self.get_output()

    def reset(self):
        if self.device == "cuda":
            dev = cp
        else:
            dev = np
        self.R = dev.zeros_like(self.U)
        self.w = dev.zeros_like(self.U)
        self.fired = dev.zeros_like(self.U).astype(bool)


class SimpleInput(InputPopulation):
    """
    A simple input population with both excitatory and inhibitory neurons.

    This class provides a convenient way to create an input population with a specified
    number of excitatory and inhibitory neurons. It combines SimpleExcitatoryInputParams
    and SimpleInhibitoryInputParams to create a mixed population.

    This is a common pattern in neural networks, where inputs can have both excitatory
    and inhibitory effects on the target neurons.
    """
    def __init__(self, n_excitatory: int, n_inhibitory: int, is_cuda: bool, is_conductive: bool):
        input_params = (SimpleExcitatoryInputParams(n_excitatory, is_cuda) +
                        SimpleInhibitoryInputParams(n_inhibitory, is_cuda))
        super().__init__(input_params, is_conductive)


class IzhPopulation(NeuronPopulation):
    """
    A population of Izhikevich neurons.

    This class implements the full Izhikevich neuron model, which can reproduce various
    firing patterns observed in real neurons. It models the membrane potential dynamics
    using the Izhikevich equations, and can also include conductance-based synapses and
    short-term plasticity (facilitation and depression).

    The Izhikevich model is defined by the following equations:
    v' = 0.04v² + 5v + 140 - u + I
    u' = a(bv - u)

    If v ≥ 30 mV, then v ← c, u ← u + d

    Where:
    - v is the membrane potential
    - u is a recovery variable
    - a, b, c, d are parameters that determine the firing pattern
    - I is the input current

    This class takes IzhParams as input to initialize the population parameters.
    Different subclasses of IzhParams can be used to create populations with different
    firing characteristics (e.g., regular spiking, fast spiking, bursting).
    """
    def __init__(self, params: IzhParams, conductive: bool,
                 tau_a: float = DEFAULT_TAU_A, tau_b: float = DEFAULT_TAU_B,
                 tau_c: float = DEFAULT_TAU_C, tau_d: float = DEFAULT_TAU_D):
        if type(params.a) == cp.ndarray:
            self.device = 'cuda'
            is_cuda = True
        else:
            is_cuda = False
            self.device = 'cpu'

        super().__init__(is_cuda, False)

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
        self.conductive = conductive

        if self.device == 'cuda':
            dev = cp
        else:
            dev = np

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

    def step(self, excitatory_input: GenArray, inhibitory_input: GenArray):
        assert isinstance(excitatory_input, type(self.v))
        assert isinstance(inhibitory_input, type(self.v))
        self.v = self.v
        self.fired = self.v >= 30
        self.v[self.fired] = self.c[self.fired]
        self.u[self.fired] = self.u[self.fired] + self.d[self.fired]

        if self.conductive:
            input_voltages = self.synaptic_current()
        else:
            input_voltages = inhibitory_input - excitatory_input

        #breakpoint()

        self.v = self.v + .5 * (.04 * self.v ** 2 + 5 * self.v + 140 - self.u - input_voltages)
        self.v = self.v.clip(max=30)
        self.v = self.v + .5 * (.04 * self.v ** 2 + 5 * self.v + 140 - self.u - input_voltages)
        self.v = self.v.clip(max=30)
        self.u = self.u + self.a * (self.b * self.v - self.u)

        if self.conductive:
            self.conductance_update(excitatory_input, inhibitory_input)
            self.facilitation_update()

    def synaptic_current(self):
        """
        Calculate the synaptic current based on conductance variables.

        This method implements a conductance-based model of synaptic transmission,
        where the current depends on the membrane potential and conductance variables.
        The model includes four types of conductance (g_a, g_b, g_c, g_d) that can
        represent different types of synaptic receptors (e.g., AMPA, NMDA, GABA_A, GABA_B).

        Returns:
            The total synaptic current to be added to the membrane potential equation.
        """
        a_term = self.g_a * self.v
        b_term = self.g_b * (((self.v + 80) / 60) ** 2) / (1 + (((self.v + 80) / 60) ** 2)) * self.v
        c_term = self.g_c * (self.v + 70)
        d_term = self.g_d * (self.v + 90)
        return a_term + b_term + c_term + d_term

    def conductance_update(self, excitatory_input: GenArray, inhibitory_input: GenArray):
        """
        Update the conductance variables based on excitatory and inhibitory inputs.

        This method implements the dynamics of synaptic conductances, which decay
        exponentially with time constants tau_a, tau_b, tau_c, tau_d, and are
        increased by excitatory and inhibitory inputs.

        The conductance variables g_a and g_b are associated with excitatory inputs,
        while g_c and g_d are associated with inhibitory inputs.

        Args:
            excitatory_input: Input from excitatory presynaptic neurons
            inhibitory_input: Input from inhibitory presynaptic neurons
        """
        #breakpoint()
        self.g_a = self.g_a + excitatory_input - self.g_a / self.tau_a
        self.g_b = self.g_b + excitatory_input - self.g_b / self.tau_b
        self.g_c = self.g_c + inhibitory_input - self.g_c / self.tau_c
        self.g_d = self.g_d + inhibitory_input - self.g_d / self.tau_d

    def facilitation_update(self):
        """
        Update the facilitation variables for short-term synaptic plasticity.

        This method implements the dynamics of short-term synaptic plasticity,
        which includes both facilitation and depression. The model is based on
        the Tsodyks-Markram model, where:
        - R represents the fraction of available synaptic resources (depression)
        - w represents the synaptic efficacy (facilitation)

        The variables are updated based on the firing state of the neurons and
        the parameters U (baseline release probability), F (facilitation time constant),
        and D (depression time constant).
        """
        self.R = self.R + (1-self.R)/self.D - self.R * self.w * self.fired
        self.w = self.w + (self.U - self.w)/self.F + self.U * (1-self.w) * self.fired

    def get_output(self):
        if self.conductive:
            return self.fired * self.w * self.R
        else:
            return self.fired

    def reset(self, voltage_range: float = 0.0):
        if self.device == 'cuda':
            dev = cp
        else:
            dev = np
        self.fired = dev.zeros_like(self.a).astype(bool)
        self.v = dev.full_like(self.a, -65.0) - dev.random.rand(*self.v.shape) * voltage_range # starting voltage
        self.u = self.b * self.v

        # conductance variables
        self.g_a = dev.zeros_like(self.a)
        self.g_b = dev.zeros_like(self.a)
        self.g_c = dev.zeros_like(self.a)
        self.g_d = dev.zeros_like(self.a)
        self.R = dev.zeros_like(self.a)
        self.w = dev.zeros_like(self.a)
