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
1. quantifying loss: how far away you are from predicted to actual

2. Loss fuction = cost function = objective function = empirical risk

3. Binary Cross Entropy Loss, can be used with model that output a probability between 0 and 1, $J(W)$

4. Mean Squared Error Loss, for continuous outputs, used with regression models

### 5. Training a Neural Networks

1. Loss Optimization, find the network weights that achieve the lowest loss

2. Loss is a function of the nework weights, Z = loss = J(w0, w1), X = w0, Y = w1, find the lowest Z with a set of (x, y)

3. Gradient Descent, until we converge

4. Computing Gradients: Backpropagation
    4.1 how does a small change of one weight to the final loss
    4.2 chain rules applied! -> backpropagation from output to input
    4.3 repeat this for every weight in the network using gradients from later layers
    
5. Loss function can be difficult to optimize
    5.1 optimization through gradient
    5.2 set the learning rate
    5.3 design an adaptive learning rates

6. Adaptive learning rates, can be made larger and smaller depends on.

7. Gradient Descent algos: Adam, SGD, Adadelta, Adagrad, RMSProp

```python
import tensorflow as tf
model = tf.keras.Sequential([...])
optimizer = tf.keras.optimizer.SGD()
while True:
    prediction = model(x)
    with tf.GradientTape() as tape:
        loss = compute_loss(y, prediction)

    # compute gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### 6. Mini Batches
> Instead of compute gradient on entire dataset on each iteration, computer over a sample of the dataset (but noisy and stochastic)

1. **Stochastic Gradient Descent**:
    1.1 Pick batch of B data points
    1.2 compute gradient on 1/B
    1.3 update weights: W <- W - a * gradient

2. Benefits: smoother covergence, more accurate estimate of gradient, allows for larger learning rates, mini-batches lead to fast training, can parallelize and can use GPU


### 7. Overfitting
1. sgd -> lead to overfitting, 
2. overfitting, too complex, extra parameter, do not generalize well
3. **regularization**: to discourage complex model, improve generalization

4. **Dropout**: randomly set some activations (neuron) to 0, turn them off

5. **Early stopping**: stop training before we have a chance of overfitting




