adahessian is the first 'second order' optimizer that actually performs (and does so extremely well) on real data.
The big drawback is you'll need to have about 2x the GPU memory that you would otherwise need to run.

The official github for adahessian is here:
https://github.com/amirgholami/adahessian

In the implementation here, I've consolidated it into a single file import instead of the util + optim file like in the official repo to make it easier to use.

Note that you have to update your training loop as below:
# usage example: 
    from adahessian import Adahessian, get_params_grad
    import torch.optim.lr_scheduler as lr_scheduler
#
    optimizer = Adahessian(model.parameters(),lr=.15)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        [30,45], # 
        gamma=.1,
        last_epoch=-1)

#
# config for training loop:
#
            loss.backward(create_graph=True)
            _, gradsH = get_params_grad(model)
            optimizer.step(gradsH)


