import math
import torch
from torch.optim.optimizer import Optimizer, required

# Original source:  DiffGrad:  https://github.com/shivram1987/diffGrad/blob/master/diffGrad.py
# RAam:  https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
# modifications: @lessw2020 - blend RAdam with DiffGrad and add version options
# __version__: 12.27.19


class diffRGrad(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 version=1,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd

        self.version = version

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(diffRGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(diffRGrad, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('diffGRad does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p_data_fp32)                    

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    state['previous_grad'] = state['previous_grad'].type_as(p_data_fp32)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                previous_grad = state['previous_grad']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1

                # compute diffgrad coefficient (dfc)

                #print("grad = ",grad.size())
                #print("prev_grad = ",previous_grad.size())

                if self.version==0:
                    diff = abs(previous_grad - grad)
                elif self.version ==1:
                    diff = previous_grad-grad
                elif self.version ==2:
                    diff =  .5*abs(previous_grad - grad)

                if self.version==0 or self.version==1:    
                    dfc = 1. / (1. + torch.exp(-diff))
                elif self.version==2:
                    dfc = 9. / (1. + torch.exp(-diff))-4      #DFC2 = 9/(1+e-(.5/g/)-4 #range .5,5

                state['previous_grad'] = grad  


                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size




                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    # update momentum with dfc
                    #print("dfc ",dfc.size())
                    #print("exp_avg ",exp_avg.size())
                    exp_avg1 = exp_avg * dfc.float()


                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg1, denom)
                    p.data.copy_(p_data_fp32)

                elif step_size > 0:

                    #print("exp_avg in elif",exp_avg.size())
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
