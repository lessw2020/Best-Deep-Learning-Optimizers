# Best-Deep-Learning-Optimizers</br>
Collection of the latest, greatest, deep learning optimizers (for Pytorch) - CNN, Transformer, NLP suitable
</br></br>
Current top performers = Have not run benchmarks lately and a lot has changed.  Quick recommendations = transformer or CNN = madgrad / adahessian.  For CNN only, Ranger. 
</br></br>
## Updates - 
April 2021:  Meet Madgrad!  </br>Have added Madgrad with an improvement to weight decay. Madgrad is a new optimizer released by FB AI in February.  In testing with transformers for image classification, madgrad blew away the various Adam variants.
However, as spotted by @nestordemeure, the weight decay impl was like adam instead of adamW.  In testing, AdamW style weight decay was the winner and thus the implementation here is with my modification to use AdamW style wd.
Recommend testing with </br>a)no weight decay, recommended by Madgrad authors and </br>b)weight decay at same level you would use for AdamW with this madgrad_wd version.
</br>
Modified madgrad is here:  https://github.com/lessw2020/Best-Deep-Learning-Optimizers/tree/master/madgrad

And original madgrad is here:  https://github.com/facebookresearch/madgrad

Pending work = there is a new paper discussing Stable Weight Decay as being the ultimate weight decay.  Planning to implement and test with madgrad soon. 

August 2020 -  AdaHessian, the first 'it really works and works really well' second order optimizer added:
 I tested AdaHessian last month on work datasets and it performed extremely well.  It's like training with a guided missile compared to most other optimizers.
The big caveat is you will need about 2x the normal GPU memory to run it vs running with a 'first order' optimizer.
I am trying to get a Titan GPU with 24GB GPU memory just for this purpose atm.


new version of Ranger with highest accuracy to date for all optimizers tested:
April 11 - New version of Ranger released (20.4.11), highest score for accuracy to date.  
</br>Ranger has been upgraded to use Gradient Centralization.  See: https://arxiv.org/abs/2004.01461  and github:  https://github.com/Yonghongwei/Gradient-Centralization

It will now use GC by default, and run it for both conv layers and fc layers. You can turn it on or off with "use_gc" at init to test out the difference on your datasets.
![](images/projected_gradient.png)
(image from gc github).   
</br>The summary of gradient centralization: "GC can be viewed as a projected gradient descent method with a constrained loss function. The Lipschitzness of the constrained loss function and its gradient is better so that the training process becomes more efficient and stable."
</br>

Note - for optimal accuracy, make sure you use run with a flat lr for some time and then cosine descent the lr (72% - 28% descent), or if you don't have an lr framework... very comparable results by running at one rate for 75%, then stop and decrease lr, and run remaining 28%. 

## Usage - GC on by default but you can control all aspects at init:
![](images/ranger-with-gc-options.jpg)
</br>
## Ranger will print settings at first init so you can confirm optimization is set the way you want it:
![](images/ranger-init.jpg)

</br> Future work: MARTHE, HyperAdam and other optimizers will be tested and posted if they look good.  

</br>
12/27 - added DiffGrad, and unofficial version 1 support (coded from the paper). 
</br>
12/28 - added Diff_RGrad = diffGrad + Rectified Adam to start off....seems to work quite well. 

Medium article (summary and FastAI example usage):
https://medium.com/@lessw/meet-diffgrad-new-deep-learning-optimizer-that-solves-adams-overshoot-issue-ec63e28e01b2

Official diffGrad paper:  https://arxiv.org/abs/1909.11015v2

12/31 - AdaMod and DiffMod added.  Initial SLS files added (but more work needed).


<b>In Progress:</b></br></br>
A - Parabolic Approximation Line Search:  https://arxiv.org/abs/1903.11991v2

B - Stochastic Line Search (SLS): pending (needs param group support)

c - AvaGrad 


<b>General papers of relevance:</b>

Does Adam stick close to the optimal point?  https://arxiv.org/abs/1911.00289v1


Probabalistic line searches for stochastic optimization (2017, matlab only but good theory work):  https://arxiv.org/abs/1703.10034v2  
