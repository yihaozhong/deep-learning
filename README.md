# deep-learning

A learning journey of deep learning

## Module

**Module 1: AI revolution, ML concepts, Introduction to ML tooling**

AI revolution and success factors; ML viewpoints: algorithmic vs. system; ML system and its
constituents; challenges in widespread AI adoption; democratization of AI; ML performance
metrics: algorithmic and system level; ML performance concepts/techniques: overfitting,
generalization, bias, variance, regularization; ML lifecycle and different stages; importance of
tooling in ML

**Module 2: Introduction to Deep Learning (DL)**

Single-layer neural network; multi-layer neural network; activation functions; loss functions;
neural network training: gradient descent, backpropagation, data preprocessing, stochastic
gradient descent, optimizers; DL hyperparameters: learning rate, batch size, momentum,
learning rate schedules; regularization techniques in DL training: dropout, early stopping, data
augmentation; contrastive loss

**Module 3: DL Training Tools and Techniques**

DL datasets: MNIST, FashionMNIST, CIFAR10/100, ImageNet; Introduction to DL frameworks:
PyTorch and Tensorflow; designing and training neural networks in Tensorflow and PyTorch; DL
training logs and their analysis; checkpointing: framework-specific support and restarting from
the checkpoint; DL stack on cloud; ML platforms on cloud: AWS, Microsoft, Google, and IBM;
DL training on cloud platforms; Learning with limited labels

**Module 4: Special Deep Learning Architectures**

Convolutional neural networks; Recurrent neural networks: RNNs, LSTMs, GRUs; Word
embeddings, Attention networks, and Transformers; Generative adversarial networks (GANs);
Few-shot learning networks; Siamese neural networks; Large Language Models (LLMs), Prompt
engineering

**Module 5: Hyperparameter Optimization and Feature Engineering**

Challenges in hyperparameter optimization; hyperparameter optimization techniques: gridsearch, random search, Bayesian; SMBO, successive halving; Hyperband algorithm; challenges
in feature engineering; tools for feature engineering: Featuretools, AutoFeat, TSFresh

**Module 6: Automated Machine Learning**

Machine learning pipeline and its automation; AutoML; Neural Architecture Search (NAS)
techniques: layer-based, cell-based, evolutionary algorithm-based, RL-based, DARTS; TAPAS
for accuracy prediction; Automated ML tools: auto-sklearn, auto-weka, Tpopt, Hyperopt-sklearn;
Neural Network Intelligence (NNI); H20 AutoML; Overview of AutoML platforms: IBM
AutoAI, Google AutoML; Open Neural Network Exchange (ONNX)

**Module 7: Robust Machine Learning**

Model brittleness; adversarial attacks; adversarial defenses; adversarial training; analysis using
Adversarial Robustness Toolbox

**Module 8: Distributed Training and Federated Learning**

Single node training with multiple GPUs; Parallelism in DL training: model parallelism, data
parallelism; synchronous and asynchronous SGD, straggler problem, stale gradients, variants of
synchronous and asynchronous SGD; Distributed DL training using Pytorch,
Pytorch data parallelism support: DataParallel, DistributedDataParallel, DistributedDataSampler,
Federated learning, FedAvg; Adversarial training in federated setting

**Module 9: Model drift and Continual learning**

ML model drift; data drift and concept drift; monitoring and drift detection; model retraining; Tools for continual learning


## Study Guide

**ML Performance Concepts**

Confusion matrix, accuracy, F1, true positive rate, false positive rate, ROC curve, overfiting, 
generalization, bias, variance, regularization (L1 and L2), logistic regression, linear separability 
 
**Deep Learning Basics**  

Single and multi-layered feedforward neural networks, activation functions, cross entropy loss, 
training and inference, forward pass, backpropagation, gradient descent, batch size, learning 
rate, learning rate decay, early stopping, dropout, batch normalization, momentum, Nesterov 
momentum,  
 
**ConvoluEonal Neural Networks**

CNN layers, padding, pooling, stride, sparse connections, parameter calculations, compute and 
memory requirements, receptive field, receptive field equivalence of stacked convolutions, 
power of small filters, contrastive loss, Siamese networks, pseudo labeling 
 
**Recurrent Neural Networks and Transformers**

RNN and LSTM architectures, parameter calculations, Transformer architecture components, 
encoder and decoder block architectures, different weight matrices in transformers, concept of 
query, key, and value, multi-head aNention, aNention calculation 
 
**Hyperparameter Optimization** 

Successive halving, Hyperband algorithm 
 
**Distributed Training**

Data parallelism, Synchronous SGD, Asynchronous SGD, Parameter Server aggregation, straggler 
problem, stale gradients, Ring all-reduce, scaling efficiency 
 
**Generative Adversarial Networks (GANs)** 

Generator, discriminator, GAN training, GAN loss function 
 
**Federated Learning** 

Federated learning challenges, Federated Averaging algorithm, problem of non-IID data  

---
<details open>
<summary><h2>Knowledge Base</h2></summary>

### ML Performance Concepts
**Confusion matrix:** A confusion matrix is a table that summarizes the performance of a classification model. It shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for each class. It helps evaluate the model's accuracy and identify any misclassifications.

**Accuracy:** Accuracy is the proportion of correct predictions made by the model out of all the predictions. It is calculated as (TP + TN) / (TP + TN + FP + FN). However, accuracy can be misleading when dealing with imbalanced datasets.

**F1 score:** The F1 score is the harmonic mean of precision and recall. Precision is the proportion of true positive predictions among all positive predictions, while recall (or true positive rate) is the proportion of true positive predictions among all actual positive instances. The F1 score provides a balanced measure of a model's performance, especially for imbalanced datasets.

**True positive rate (TPR) and false positive rate (FPR):** TPR, also known as recall or sensitivity, is the proportion of actual positive instances that are correctly identified by the model. FPR, on the other hand, is the proportion of actual negative instances that are incorrectly classified as positive. These metrics are used to evaluate the model's ability to identify positive instances while minimizing false alarms.

**ROC curve:** The Receiver Operating Characteristic (ROC) curve is a graphical representation of a binary classifier's performance at various classification thresholds. It plots the TPR against the FPR. The area under the ROC curve (AUC-ROC) is a measure of the model's discriminatory power, with a higher AUC indicating better performance.

**Overfitting:** Overfitting occurs when a model learns the noise and peculiarities of the training data too well, resulting in poor performance on unseen data. An overfit model has high variance and low bias, meaning it is too complex and fails to generalize well.

**Generalization:** Generalization refers to a model's ability to perform well on unseen data. A model that generalizes well can make accurate predictions on new, previously unseen instances.

**Bias and variance:** Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias models are too simplistic and underfit the data. Variance, on the other hand, refers to the model's sensitivity to small fluctuations in the training data. High variance models are too complex and overfit the data.

**Regularization (L1 and L2):** Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. L1 regularization (Lasso) adds the absolute values of the model's coefficients to the loss, encouraging sparsity. L2 regularization (Ridge) adds the squared values of the coefficients, encouraging smaller weights overall. Both techniques help control model complexity and improve generalization.

**Logistic regression:** Logistic regression is a classification algorithm that models the probability of an instance belonging to a particular class. It uses the logistic (sigmoid) function to map the input features to a probability value between 0 and 1. The decision boundary in logistic regression is linear.

**Linear separability:** A dataset is said to be linearly separable if there exists a hyperplane that can perfectly separate the instances of different classes. In such cases, a linear classifier like logistic regression can achieve perfect classification.

### Deep Learning Basics
 
**Single and multi-layered feedforward neural networks:** Feedforward neural networks are the simplest type of artificial neural networks. They consist of an input layer, one or more hidden layers, and an output layer. In a single-layer feedforward network, there is only one hidden layer, while in a multi-layer network, there are multiple hidden layers. Each layer is fully connected to the next layer, meaning each neuron in one layer is connected to every neuron in the following layer. Information flows in one direction, from the input layer through the hidden layers to the output layer, without any loops or feedback connections.

**Activation functions:** Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns and relationships in the data. They are applied to the weighted sum of inputs at each neuron to determine its output. Some common activation functions include:

1. Sigmoid: Squashes the input to a value between 0 and 1.
2. Tanh: Squashes the input to a value between -1 and 1.
3. ReLU (Rectified Linear Unit): Returns 0 for negative inputs and the input itself for positive values.
4. Leaky ReLU: Similar to ReLU but allows small negative values to pass through.

**Cross-entropy loss:** Cross-entropy loss is a commonly used loss function for classification tasks. It measures the dissimilarity between the predicted class probabilities and the true class labels. The goal is to minimize the cross-entropy loss during training, which encourages the model to assign high probabilities to the correct classes. For binary classification, binary cross-entropy is used, while for multi-class classification, categorical cross-entropy is employed.

**Training and inference:** Training a neural network involves adjusting its weights to minimize the loss function on the training data. This is done through an optimization algorithm like gradient descent. During training, the network makes predictions on the input data (forward pass), computes the loss, and then backpropagates the gradients to update the weights (backward pass). Inference, on the other hand, is the process of using a trained model to make predictions on new, unseen data. During inference, only the forward pass is performed.

**Forward pass and backpropagation:** The forward pass involves computing the outputs of each layer in the network, given the input data. The input is multiplied by the weights of the first layer, passed through an activation function, and then fed as input to the next layer. This process continues until the output layer is reached. Backpropagation is the process of computing the gradients of the loss function with respect to each weight in the network. It starts from the output layer and propagates the gradients backward through the network, using the chain rule to compute the gradients at each layer.

**Gradient descent:** Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the weights of the network. It works by computing the gradients of the loss function with respect to the weights and updating the weights in the direction of steepest descent. The learning rate determines the size of the steps taken in the weight space during each update. Common variations of gradient descent include batch gradient descent (which uses the entire training set), mini-batch gradient descent (which uses subsets of the training data), and stochastic gradient descent (which uses individual training examples).

**Batch size:** The batch size refers to the number of training examples used in one iteration of gradient descent. It determines the number of samples over which the gradients are computed and the weights are updated. Smaller batch sizes lead to more frequent weight updates and can help escape local minima, but they also introduce more noise in the gradients. Larger batch sizes provide more stable gradient estimates but may converge more slowly and require more memory.

**Learning rate and learning rate decay**: The learning rate is a hyperparameter that controls the step size at which the weights are updated during gradient descent. A high learning rate can cause the optimization to overshoot the minimum, while a low learning rate may result in slow convergence. Learning rate decay is a technique where the learning rate is gradually reduced over the course of training. This allows the optimizer to take larger steps initially and then fine-tune the weights as it approaches the minimum.

**Early stopping:** Early stopping is a regularization technique used to prevent overfitting. It involves monitoring the model's performance on a validation set during training and stopping the training process when the performance on the validation set starts to degrade. This helps to avoid overfitting to the training data and ensures that the model generalizes well to unseen data.

**Dropout:** Dropout is another regularization technique that helps prevent overfitting. During training, dropout randomly sets a fraction of the input units to zero at each update. This prevents the neurons from co-adapting and forces them to learn more robust features. Dropout acts as an ensemble technique, as it trains multiple subnetworks with shared weights, which are combined during inference.

**Batch normalization:** Batch normalization is a technique used to normalize the activations of each layer in the network. It helps to stabilize the training process and reduce the sensitivity to the choice of initialization. Batch normalization works by computing the mean and variance of the activations within a mini-batch and normalizing them to have zero mean and unit variance. It then applies a learnable scale and shift to the normalized activations.

**Momentum and Nesterov momentum:** Momentum is a technique used to accelerate gradient descent by adding a fraction of the previous update to the current update. It helps to smooth out oscillations and converge faster. Nesterov momentum is a variant of momentum that looks ahead in the direction of the previous update and corrects the gradient based on the anticipated future position. Nesterov momentum often leads to faster convergence compared to standard momentum.
 
 
</details>