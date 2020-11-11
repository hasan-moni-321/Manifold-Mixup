 ## Kannada-MNIST
 
 ### 1. Manifold Mixup Using PyTorch
 Here I implement the idea from the paper [Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/pdf/1806.05236.pdf).
 It is a new regularization method which allows for training very deep and wide neural networks with much less overfitting.
 
 #### Input mixup

 It's easier to first understand what is input mixup: [mixup: BEYOND EMPIRICAL RISK MINIMIZATION.](https://arxiv.org/pdf/1710.09412.pdf)
 Input mixup is a regularization done during training procedure.                                                                                                 
 In short: model is given a linear combination of inputs and is asked to return linear combination of outputs.
 It forces the network to interpolate between samples.
 
 #### Manifold mixup

 Manifold mixup is a similar idea, but the interpolation is done at a random layer inside neural network.
 Sometimes it is the 0'th layer, which means input mixup.
 It forces the network to interpolate between hidden representations of samples.

