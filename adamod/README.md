AdaMod is a new optimizer that takes Adam but adds an exponential moving average of the adaptive learning rates. 
This ensures no large spikes during training and helps achieve faster and better convergence.

Original source code and paper:  https://github.com/lancopku/AdaMod

DiffMod is a combination of DiffGrad + AdaMod = diffgrad.

Currently DiffMod, using version 0 of DiffGrad, appears to be the best performer of all.  But more testing is needed.</br>

Usage:</br>
from diffmod import DiffMod</br>
optar = partial(DiffMod,version=0)</br>
learn = Learner(data, model, metrics=[accuracy], wd=1e-3,</br>
                opt_func=optar,</br>
                 bn_wd=False, true_wd=True,</br>
                loss_func = LabelSmoothingCrossEntropy()) </br>
