import torch
import torch.cuda

import numpy as np
#import contextlib


def check_armijo_conditions(step_size, step_size_old, loss, grad_norm,
                      loss_next, c, beta_b):
    found = 0

    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)

    if (break_condition <= 0):
        found = 1

    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b

    return found, step_size, step_size_old

def check_goldstein_conditions(step_size, loss, grad_norm,
                          loss_next,
                          c, beta_b, beta_f, bound_step_size, eta_max):
	found = 0
	if(loss_next <= (loss - (step_size) * c * grad_norm ** 2)):
		found = 1

	if(loss_next >= (loss - (step_size) * (1 - c) * grad_norm ** 2)):
		if found == 1:
			found = 3 # both conditions are satisfied
		else:
			found = 2 # only the curvature condition is satisfied

	if (found == 0):
		raise ValueError('Error')

	elif (found == 1):
		# step-size might be too small
		step_size = step_size * beta_f
		if bound_step_size:
			step_size = min(step_size, eta_max)

	elif (found == 2):
		# step-size might be too large
		step_size = max(step_size * beta_b, 1e-8)

	return {"found":found, "step_size":step_size}


def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1,
               init_step_size=None):
    if reset_option == 0:
        pass

    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)

    elif reset_option == 2:
        step_size = init_step_size

    return step_size

def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        p_next.data = p_current - step_size * g_current

def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]

#@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

#@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state, device)