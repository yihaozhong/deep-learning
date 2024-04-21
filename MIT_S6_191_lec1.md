# Introduction to deep learning

## Lecture 1: introduction

### 1. Why deep learning, why now

1. learn the pattern underneath data
2. data is parrallizable
3. hardware advance and open source tool

### 2. Perceptron
> The structural building block of deep learning, a single neuron

1. Takes **inputs**, through weights multiplicating (w), and add up to be a **sum**, add to a non-linear function, and let to output

2. add a **bias** $w_0$, $y = g(w_0 + X^TW)$ and g is a sigmoid function $g(z) = \frac{1}{1+e^{-z}}$

3. others are sigmoid, hyperbolic tangent, Rectified Linear Unit (ReLU)

4. Why need a activation functions
    4.1 to introduce non-linearities to the network
    4.2 non-linearities allow us to deal with complex functions and non-linear data
    EXP: $g(1+3x_2+x_2)$, inside is a 2D line 

5. three steps, multiple + sum + add non-linearity


### 3. Building NN using Perceptron
1. dot product, add bias, non-linearity

2. $z_i = w_{0, i} + \sum_{j=1}^m{x_jw_{j, i}}$

3. **Dense** layers: all inputs are connected to all outputs

4. Single layer NN: input - one hidden layer - output

5. Multi output Perceptron (stacking up)
  
### 4. Application of NN
