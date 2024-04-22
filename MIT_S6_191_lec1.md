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


## Lecture 2: Recurrent Neural Networks, Transformers, and Attention

### 1. Deep Sequential Model

1. Text, Words, etc
2. One to one: binary classfication (student -> pass or not)
3. Many to one: sentiment classfication
4. One to many: image captioning, text generation
5. Many to many: machine translation, forecasting, music generation

### 2. Neurons with Recurrence
> How we links neurons with different timestamps via replica

1. feed-forward network revisited
    1.1 handling individual time steps
    1.2 try to handle input coming in with different time
    1.3 $y_t = f(x_t)$ at a time $t$

2. Neurons with Recurrence
    2.1 define $h_t$ as past memory, and $y_t = f(x_t, h_{t-1})$ at a time $t$
    2.2 recurrent cell -> loop that fed back into the neurons
    2.3 temporal cyclic dependency

3. **RNNs**
    3.1 RNN have a state, $h_t$, updated at each time step as a sequence is processed
    3.2 apply a recurrence relation at every time stamp
    3.3 cell state $h_t = f_w(x_t, h_{t-1})$ 
    3.4 same function and set of param are used at every timestep
    3.5 generate output prediction + a update hidden state

4. RNN State Update and output
    4.1 Input Vector $x_t$
    4.2 Update Hidden State, $h_t = tanh(W^T_h h_{t-1} + W^T_x x_t)$
    4.3 Output Vector $y_t = W^T_h h_t$

5. Three weight: $W_{xh}$ that update the hidden state from input dim, $W_{hh}$ that update from previous steps to next RNN unit, $W_{hy}$ that update the output dim

6. RNNs for Sequence model: one to one, many to one, one to many, many to many

7. To model seq, we **need to**:
    7.1 handle variable-length seq
    7.2 track long term dependencies
    7.3 mantain information about order
    7.4 share param across seq

### 3. A sequence modeling problem: predict the next word
>Given some words, predict the next word

1. First step: represent the English language to a NN
    1.1 require numerical inputs
    1.2 **Embedding**: transform indexes into a vector of fixed size
    1.3 Vocabulary: corpus of words

2. **Embedding**:
    2.1 Vocabulary
    2.2 indexing: word to index
    2.3 Embedding: index to fixed sized vector
        a. One-hot embedding "cat" = [0,1,0,0,0,0] , cat = 1 (index)

### 4. Backpropagation Through Time (BPTT)

1. Tricky to implement: computing the gradient wrt $h_0$ involves many factors of $W_{hh}$ + repeated gradient computation. 

2. exploding gradients: many values > 1, using gradient clipping

3. vanishing gradients, many values < 1
    3.1 use activation function
    3.2 weight initialization: initialize weights to identity matrix, initialize bias to be zero
    3.3 network architecture: use **gates** to selecttively add or remove info with each rcurrent unit (optionally let through information)

4, **LSTM**: forget, store, update, output
    4.1 maintain a cell state, use gates to control the flow of info 
        (forget gate get rid of irrelevant information)
        (Store relevant info from current input)
        (selectively update cell state)
        (output gate return a filtered version of the cell state)

### 5. RNN Application & Limits

1. limitations: encoding bottleneck
    1.1 processed timestamp by timestamp
    1.2 how to make sure all info are encoded and process
2. limitations: slow, no parallelization
3. limitations: Not very long memory, not even LSTM

4. **Desired Capabilities**
    4.1 Continous stream
    4.2 Parallelization
    4.3 Long Memory

5. Can we feed everyting into dense network: no recurrence, no order, no long memory, not scalable?

    5.1 Key idea: identify and attend to what's important

### 6. Attention is All You Need
> Attending to the most important parts of an input
1. **Self-Attention**
    1.1 identify which parts to attend to, extract the features with highest attention (similar to a search problem)

    1.2 Internet search -> **Query (Q)** entered -> search output **Key (k1)**, Key (k2)... how similar is the ke (1,2,3...) to the query -> Extract the **Value (V)** corresponding to the key

2. Learning Self Attention with NN
    a. Encode position information (in terms of order), position-aware embedding
    b. Extract query, key, value for search
        b.1 Positional embedding x linear layer = Output (Query)
        b.2 Positional embedding x linear layer (different layer) = Output (Key)
        b.3 Positional embedding x linear layer (different layer)= Output (Value)

    c. Compute attention weight (score), compute pairwise similarity between each query and key -- cosine similarity -- dot product and scale -- softmax$(\frac{Q  K^T}{scaling})$ = attention weight

    d. extract features with high attention: *attention score x value* = output --> softmax$(\frac{Q  K^T}{scaling})$ *matmul* V = A(Q, K, V)

