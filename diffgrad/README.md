DiffGrad adjusts the step size for each parameter by comparing the current gradient vs the previous.  It is designed to solve the 'Adam' 
overshoot problem, where the momentum of Adam can carry it right over the global mininimum.

https://github.com/shivram1987/diffGrad  for original source 

and paper:  https://arxiv.org/abs/1909.11015v2

(TF version - if you are forced to use TF, here's a TF version of diffgrad:
https://github.com/evanatyourservice/diffGrad-tf )


This version adds in a version parameter:  version 0 is the main one used in the paper.  version 1 removes the abs value from the calculations and
allows faster clamping.
Use:  version=1 in your optimizer params.  version=0 is default.

12/27 - added DiffRGrad - this is diffGrad with Rectified Adam to start.  Thus no warmup needed and diffGrad kicks in after Rectified Adam says variance is ready to go. 

Medium article and example usage:  https://medium.com/@lessw/meet-diffgrad-new-deep-learning-optimizer-that-solves-adams-overshoot-issue-ec63e28e01b2
