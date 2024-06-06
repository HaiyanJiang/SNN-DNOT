from typing import Callable, overload
import torch
import torch.nn as nn
from . import surrogate
from .neuron_spikingjelly import IFNode, LIFNode




def safe_divide(numerator, denominator, default_value=1.0):
    # Check for NaN or Infinite values
    mask = torch.isnan(numerator) | torch.isnan(denominator) | torch.isinf(numerator) | torch.isinf(denominator)
    result = torch.where(mask, default_value, numerator / denominator)
    mask2 = torch.isnan(result) | torch.isinf(result) | torch.eq(denominator, 0.0)
    result2 = torch.where(mask2, default_value, result)
    return result2



class OnlineIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = None,
            surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True,
            track_rate: bool = True, neuron_dropout: float = 0.0,
            track_spike: bool = True, **kwargs):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        self.track_spike = track_spike
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)
        if self.track_spike:
            self.register_memory('spike', None)


    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)


    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        # save_spike = kwargs.get('save_spike', False)
        save_spike = kwargs.get('save_spike', True)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike


class OnlineLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True,
            track_rate: bool = True, neuron_dropout: float = 0.0,
            track_spike: bool = True, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        self.track_spike = track_spike
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)
        if self.track_spike:
            self.register_memory('spike', None)


    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x


    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)


    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        # # save_spike = kwargs.get('save_spike', True)  # modified code.
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        # ### case 0: raw settings
        # self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        # self.neuronal_reset(spike)

        # # # ### case 8: in OTMD_cutout_8_SNN # torch.clamp(pt / pk, min=-(1-1./self.tau), max=1-1./self.tau)  #
        self.v_old = self.v.clone().detach()  # ## new added
        if self.spike is not None:
            self.s_old = self.spike.clone().detach()  # ## new added
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        self.v_new = self.v.clone().detach()  # ## new added
        a = -(1-1./self.tau)
        if self.spike is None:
            print('=== self.spike is None ')
            self.Pkt = torch.clamp(self.v_new - self.v_threshold * spike.clone().detach(), min=a, max=1-1./self.tau)
        elif self.s_old.shape == spike.shape:
            print('=== self.s_old.shape == spike.shape ')
            pt = self.v_new - self.v_threshold * spike.clone().detach()
            pk = self.v_old - self.v_threshold * self.s_old
            self.Pkt = torch.clamp(safe_divide(pt, pk, default_value=1-1./self.tau), min=a, max=1-1./self.tau)   # ## safe_divide(px, pu, default_value=0.0)  # ## px / pu
            # ###### without torch.clamp(), there will be nan, inf, after pt / pk.
        else:
            print('=== else others ')
            self.Pkt = torch.clamp(self.v_new - self.v_threshold * spike.clone().detach(), min=a, max=1-1./self.tau)
        # ## print(f'*** Pkt: {self.Pkt}')
        # print(f'*** Pkt.mean: {torch.mean(self.Pkt)}; (1-1./self.tau): {1 - 1. / self.tau} ***')
        # print(f'*** Pkt.shape: {self.Pkt.shape}; spike.shape: {spike.shape} ***')
        # print(f'*** spike rate: {torch.mean(1.0*spike)}; spike sum: {torch.sum(1.0*spike)} ***')


        # #########
        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        self.spike = spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    # ## self.rate_tracking = self.rate_tracking * (1-1./self.tau) + spike.clone().detach()
                    self.rate_tracking = self.rate_tracking * self.Pkt + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike