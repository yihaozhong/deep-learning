# Introduction to deep learning

## Lecture 2: CNN

### 1. What computer see?

1. Images are numbers -> grey scale matrix of number

2. Feature extraction and detect features to classify (not robust due to variation)

3. Learn a hierarchy level of features

### 2. Learning Visual Features (images)

1. how can we use spatial structure of image?

2. Idea: connect patches of input to neurons in hidden layer (a small subset)

3. Using Spatial Structure: a sliding windows to define connections

4. how can we weigt the patches

5. Patchy operation = convolution

6. Patch -> filter: use multiple filters to extract different features

### 3. Feature extraction and convolutional: a case study
 
1. feature of 'X': patch of subset -> roughly same patches

2. filter: elementwise multiply, to get a output feature map

3. Producing feature mapping

4. define filters -> training the filters as weights

### 4. CNN

1. **Convolution** apply filters to generate feature maps
2. **Non-linearity**: Often ReLU (activating on the result of feature maps)
3. **Pooling:** downsampling operation on each feature map
4. **feed result** to a fully connected layer

5. For a neuron in hidden layer (layers.Conv2D):
    5.1 take inputs from patch
    5.2 compute weighted sum (linear combination)
    5.3 apply bias
    
6. **Conv2D(filters = d, kernel_size = (h, w), strides = s)**
    6.1 Layer dimensions: height, width, depth, where $h$ and $w$ are spatial dimensions and $d$ = number of filters
    6.2 Stride: filter step size 
    6.3 Receptive Field: locations in input image that a node is path connected to

7. **Non-linearity**
    7.1 apply after every convolution operation (after convolutional layer)
    7.2 ReLU: pixel by pixel operation that replace all negative values by zero
    7.3 $g(z) = max(0, z)$ Recitified Linear Unit (ReLU)

8. **Pooling: MaxPool2D(pool_size= (2, 2), strides =2)**
    8.1 reduced dimensionality and still preserve spatial invariance
    8.2 simply takes the maximum of the 2*2 filters

9. **INPUT -> FEATURE LEARNING(Conv/ReLU/Pooling) -> FLATTEN -> CLASSIFICATION (FULLY CONNECTED NN + SOFTMAX)**
```python
model = Sequential(
    layers.Conv2D(ReLu)
    layers.MaxPool2D

    layers.Conv2D(ReLu)
    layers.MaxPool2D

    layers.Flatten()
    Dense(1024, ReLu)
    Dense(10, softmax)
)
```

### 5. Architecture for many applications
1. After the feature learning, we can do classification/object detection/segmentation/probabilistic control

2. R-CNN: find regiond that we think have objects and use CNN to classify (Region-based), use selective search to find out region

3. Faster R-CNN learns regions proposals
    3.1 Region Proposal Network