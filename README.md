# deep-learning

A learning journey of deep learning

Module 1: AI revolution, ML concepts, Introduction to ML tooling

AI revolution and success factors; ML viewpoints: algorithmic vs. system; ML system and its
constituents; challenges in widespread AI adoption; democratization of AI; ML performance
metrics: algorithmic and system level; ML performance concepts/techniques: overfitting,
generalization, bias, variance, regularization; ML lifecycle and different stages; importance of
tooling in ML

Module 2: Introduction to Deep Learning (DL)

Single-layer neural network; multi-layer neural network; activation functions; loss functions;
neural network training: gradient descent, backpropagation, data preprocessing, stochastic
gradient descent, optimizers; DL hyperparameters: learning rate, batch size, momentum,
learning rate schedules; regularization techniques in DL training: dropout, early stopping, data
augmentation; contrastive loss

Module 3: DL Training Tools and Techniques

DL datasets: MNIST, FashionMNIST, CIFAR10/100, ImageNet; Introduction to DL frameworks:
PyTorch and Tensorflow; designing and training neural networks in Tensorflow and PyTorch; DL
training logs and their analysis; checkpointing: framework-specific support and restarting from
the checkpoint; DL stack on cloud; ML platforms on cloud: AWS, Microsoft, Google, and IBM;
DL training on cloud platforms; Learning with limited labels

Module 4: Special Deep Learning Architectures

Convolutional neural networks; Recurrent neural networks: RNNs, LSTMs, GRUs; Word
embeddings, Attention networks, and Transformers; Generative adversarial networks (GANs);
Few-shot learning networks; Siamese neural networks; Large Language Models (LLMs), Prompt
engineering

Module 5: Hyperparameter Optimization and Feature Engineering

Challenges in hyperparameter optimization; hyperparameter optimization techniques: gridsearch, random search, Bayesian; SMBO, successive halving; Hyperband algorithm; challenges
in feature engineering; tools for feature engineering: Featuretools, AutoFeat, TSFresh

Module 6: Automated Machine Learning

Machine learning pipeline and its automation; AutoML; Neural Architecture Search (NAS)
techniques: layer-based, cell-based, evolutionary algorithm-based, RL-based, DARTS; TAPAS
for accuracy prediction; Automated ML tools: auto-sklearn, auto-weka, Tpopt, Hyperopt-sklearn;
Neural Network Intelligence (NNI); H20 AutoML; Overview of AutoML platforms: IBM
AutoAI, Google AutoML; Open Neural Network Exchange (ONNX)

Module 7: Robust Machine Learning

Model brittleness; adversarial attacks; adversarial defenses; adversarial training; analysis using
Adversarial Robustness Toolbox

Module 8: Distributed Training and Federated Learning

Single node training with multiple GPUs; Parallelism in DL training: model parallelism, data
parallelism; synchronous and asynchronous SGD, straggler problem, stale gradients, variants of
synchronous and asynchronous SGD; Distributed DL training using Pytorch,
Pytorch data parallelism support: DataParallel, DistributedDataParallel, DistributedDataSampler,
Federated learning, FedAvg; Adversarial training in federated setting

Module 9: Model drift and Continual learning

ML model drift; data drift and concept drift; monitoring and drift detection; model retraining;
Tools for continual learning