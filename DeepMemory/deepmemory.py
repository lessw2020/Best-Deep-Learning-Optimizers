import math
import torch
from torch.optim import Optimizer

# DeepMemory is designed to offset the weakness of many adaptive optimizers by creating a 'long term' memory of the gradients over the course of an epoch.
# This long term memory is averaged against the current adaptive step size generated from the current mini-batch in order to help guide the step size more optimally.

# DeepMemory also keeps a short term gradient buffer that was developed in diffgrad, and locks down the step size when minimal gradient change is detected.

# 1/1/2020 - @lessw2020 developed the long term memory concept as a blended average (vs max throttle in AdaMod), and created and tested deep Memory
# credits:
# DiffGrad:  Uses the local gradient friction clamp developed by DiffGrad, but with version 1 coded by lessw from the paper:
# https://github.com/shivram1987/diffGrad (S.R.Dubey et al)

# AdaMod - DeepMemory builds on the concepts for longer term monitoring in AdaMod (b3 concept but changed from min throttling to blended average and changed input to len_memory and size):

# AdaMod source and paper link - https://github.com/lancopku/AdaMod/blob/master/adamod/adamod.py

# modifications @lessw2020
# 1/1/20 = instead of b3, change to 'len_memory' and compute b3 (.99 is really 100 memory as 1-(1/100)= .99)


class DeepMemory(Optimizer):
    """Implements DeepMemory algorithm (built upon DiffGrad and AdaMod concepts) with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        len_memory = b3 (smoothing coefficient from AdaMod) in easier to use format, mem average with b3 is averaged with immmediate gradient.  
            specify the memory len, b3 is computed.
        version = 0 means .5 clamping rate, 1 = 0-1 clamping rate (from DiffGrad)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=4e-3, betas=(0.9, 0.999), len_memory=200, version=1,
                 eps=1e-6, weight_decay=0, debug_print=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        #compute b3
        base = 1/len_memory
        beta3 = 1-(base)
        print(f"DeepMemory: length of memory is {len_memory} - this should be close or equal to batches per epoch")
        
        #debugging
        self.debug_print=debug_print
        
        
        if not 0.0 <= beta3 < 1.0:
            raise ValueError("Invalid len_memory parameter: {}".format(beta3))
        
        defaults = dict(lr=lr, betas=betas, beta3=beta3, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.version = version

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'DiffMod does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of actual learning rates
                    state['exp_avg_lr'] = torch.zeros_like(p.data)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p.data)                    
                    

                exp_avg, exp_avg_sq, exp_avg_lr = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_lr']
                previous_grad = state['previous_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # compute diffgrad coefficient (dfc)
                if self.version==0:
                    diff = abs(previous_grad - grad)
                    
                elif self.version ==1:
                    diff = previous_grad-grad
               
                    
                if self.version==0 or self.version==1:    
                    dfc = 1. / (1. + torch.exp(-diff))
                
                    
                state['previous_grad'] = grad                

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # create long term memory of actual learning rates (from AdaMod)
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom)
                exp_avg_lr.mul_(group['beta3']).add_(1 - group['beta3'], step_size)
                
                if self.debug_print:
                    print(f"batch step size {step_size} and exp_avg_step {exp_avg_lr}")
                    
                #Blend the mini-batch step size with long term memory
                step_size = step_size.add(exp_avg_lr)
                step_size = step_size.div(2.)
                    
                
                # update momentum with dfc
                exp_avg1 = exp_avg * dfc
                
                step_size.mul_(exp_avg1)

                p.data.add_(-step_size)

        return loss