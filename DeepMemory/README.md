DeepMemory is a new optimizer I came up with after blending DiffGrad + AdaMod.  The core concept is to provide the optimizer with
long term memory of the previous step sizes. 

Results in initial testing put it on par with Ranger and both Ranger and DeepMemory topped the recent testing I did with about 8 different optimizers.


DeepMemory is designed to offset the weakness of many adaptive optimizers by creating a 'long term' memory of the gradients over the course of an epoch.
This long term memory is averaged against the current adaptive step size generated from the current mini-batch in order to help guide the step size more optimally.

DeepMemory also keeps a short term gradient buffer that was developed in diffgrad, and locks down the step size when minimal gradient change is detected.

1/1/2020 - @lessw2020 developed the long term memory concept as a blended average (vs max throttle in AdaMod), and created and tested deep Memory
credits:
DiffGrad:  Uses the local gradient friction clamp developed by DiffGrad, but with version 1 coded by lessw from the paper:
https://github.com/shivram1987/diffGrad (S.R.Dubey et al)

AdaMod - DeepMemory builds on the concepts for longer term monitoring in AdaMod (b3 concept but changed from min throttling to blended average and changed input to len_memory and size):

AdaMod source and paper link - https://github.com/lancopku/AdaMod/blob/master/adamod/adamod.py

modifications @lessw2020
1/1/20 = instead of b3, change to 'len_memory' and compute b3 (.99 is really 100 memory as 1-(1/100)= .99)
