adahessian is the first 'second order' optimizer that actually performs (and does so extremely well) on real data.
The big drawback is you'll need to have about 2x the GPU memory that you would otherwise need to run.

The official github for adahessian is here:
https://github.com/amirgholami/adahessian

In the implementation here, I've consolidated it into a single file import instead of the util + optim file like in the official repo to make it easier to use.

