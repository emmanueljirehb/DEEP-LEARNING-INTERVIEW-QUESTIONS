# Deep Learning Questions and Answers

Here are concise answers to common deep learning questions.

---

### 1. What is Neural Networks?
Neural networks are computational models inspired by the human brain, designed to recognize patterns. They consist of interconnected layers of nodes (neurons) that process data, learn from it, and make predictions or classifications without explicit programming.

### 2. What are Weights?
Weights are parameters within a neural network that determine the strength of the connection between two neurons. During training, these weights are adjusted to learn patterns and minimize the difference between predicted and actual outputs.

### 3. What is a Layer, Node, Connections?
A **layer** is a collection of interconnected nodes in a neural network. A **node** (or neuron) is a computational unit that receives inputs, processes them, and passes an output. **Connections** are the links between nodes across different layers, each associated with a specific weight.

### 4. What is Dense Layer?
A dense (or fully connected) layer is a type of neural network layer where every neuron in the layer is connected to every neuron in the preceding layer. It's commonly used at the end of a network for classification or regression tasks.

### 5. Why to Introduce Activation/Non-Linear Functions?
Activation functions introduce non-linearity into the network, allowing it to learn complex patterns and relationships in data that linear models cannot. Without them, a neural network, no matter how deep, would behave like a single-layer perceptron.

### 6. What is the Purpose of Loss/Cost/Error Functions?
Loss functions quantify the difference between the predicted output of a neural network and the actual target value. They provide a measure of how well the model is performing, which is then used to guide the model's learning process during optimization.

### 7. What is the Purpose of Optimizers?
Optimizers are algorithms or methods used to adjust the weights and biases of a neural network during training to minimize the loss function. They determine how the model updates its internal parameters to learn from the errors it makes.

### 8. Name Few Activation Functions.
Sigmoid, Tanh, ReLU (Rectified Linear Unit), Leaky ReLU, Softmax.

### 9. Name Few Optimizers.
Stochastic Gradient Descent (SGD), Adam, RMSprop, Adagrad, Adadelta.

### 10. Name Few Cost Functions.
Mean Squared Error (MSE), Mean Absolute Error (MAE), Binary Cross-Entropy, Categorical Cross-Entropy.

### 11. Explain Gradient Descent Optimizer.
Gradient Descent is an iterative optimization algorithm that adjusts model parameters (weights and biases) in the direction opposite to the gradient of the loss function. It aims to find the minimum of the loss function by taking small steps proportional to the negative of the gradient.

### 12. Explain Stochastic Gradient Descent Optimizer.
Stochastic Gradient Descent (SGD) is a variant of Gradient Descent where model parameters are updated after processing *each individual training example*, rather than the entire dataset. This makes updates more frequent and can help escape shallow local minima.

### 13. Explain Adam (Adaptive Momentum) Optimizer.
Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the concepts of momentum and RMSprop. It adaptively calculates learning rates for each parameter based on the first (mean) and second (uncentered variance) moments of the gradients.

### 14. Explain Adagrad (Adaptive Gradient) Optimizer.
Adagrad adapts the learning rate for each parameter based on the historical sum of squared gradients. It performs larger updates for infrequent parameters and smaller updates for frequent parameters, useful for sparse data.

### 15. What is Local Minima & Global Minima?
A **global minimum** is the point where the loss function has its lowest possible value across the entire parameter space. A **local minimum** is a point where the loss function is lower than in its immediate vicinity, but not necessarily the lowest overall.

### 16. Explain Difference Between Sigmoid, Tanh Activation Functions.
Sigmoid squashes outputs between 0 and 1, making it suitable for binary classification outputs. Tanh (hyperbolic tangent) squashes outputs between -1 and 1, centering the output around zero, which can aid in faster convergence in some cases.

### 17. Explain Difference Between ReLU, Leaky ReLU Activation Functions.
ReLU outputs the input directly if positive, and zero otherwise, addressing the vanishing gradient problem. Leaky ReLU is a variant that allows a small, non-zero gradient when the input is negative, preventing "dying ReLUs" where neurons become inactive.

### 18. Explain the Difference Between Step, ReLU Activation Functions.
A Step function (or Heaviside function) is a threshold-based activation that outputs 0 for inputs below a certain value and 1 (or another constant) for inputs above it, acting as a binary switch. ReLU provides a continuous output for positive inputs, allowing for more nuanced gradient flow.

### 19. When to Use ReLU, Tanh, Sigmoid Linear Activation Functions?
**ReLU** is typically preferred for hidden layers in deep networks due to its computational efficiency and ability to mitigate vanishing gradients. **Sigmoid** is often used in the output layer for binary classification problems. **Tanh** can be used in hidden layers as an alternative to sigmoid, especially when outputs centered around zero are beneficial.

### 20. What are the Cost Functions for Regression Problem?
Common cost functions for regression include Mean Squared Error (MSE), which penalizes larger errors more heavily, and Mean Absolute Error (MAE), which measures the average magnitude of errors without considering their direction.

### 21. What are the Cost Functions for Classification Problem?
Common cost functions for classification include Binary Cross-Entropy for binary classification, and Categorical Cross-Entropy (or Sparse Categorical Cross-Entropy) for multi-class classification problems.

### 22. What are the Activation Functions for Regression Problem?
For regression problems, the output layer typically uses a **linear (no) activation function**, allowing the model to output any real number. If outputs are constrained (e.g., between 0 and 1), sigmoid or other limited range activations might be used.

### 23. What are the Activation Functions for Classification Problem?
For binary classification, **Sigmoid** is used in the output layer. For multi-class classification, **Softmax** is used in the output layer to produce a probability distribution over the classes.

### 24. What is the Purpose of Convolution Layer in CNN?
The convolution layer in a CNN applies learnable filters (kernels) to the input data to detect local patterns such as edges, textures, or shapes. It extracts relevant features by performing a convolution operation, creating feature maps.

### 25. What is the Purpose of Max Pooling Layer in CNN?
The max pooling layer down-samples the feature maps, reducing their spatial dimensions (height and width) while retaining the most important features (the maximum values). This helps reduce computational cost, memory usage, and makes the detected features more robust to slight translations.

### 26. What is the Purpose of Flatten Layer in CNN?
The flatten layer converts the multi-dimensional output of convolutional and pooling layers into a 1D vector. This is necessary to transition the data from the convolutional part of the CNN to the fully connected (dense) layers that typically perform classification or regression.

### 27. What is the Purpose of Dropout in CNN or NN?
Dropout is a regularization technique that randomly sets a fraction of neurons' outputs to zero during training. This prevents complex co-adaptations between neurons, forces the network to learn more robust features, and reduces overfitting.

### 28. How to Choose Epochs in NN Training?
The number of epochs is chosen empirically, often by monitoring the validation loss during training. You typically train until the validation loss stops decreasing or starts increasing (indicating overfitting), a technique known as early stopping.

### 29. How Images or Photos Represented in Code (DataFrames)?
Images are typically represented as multi-dimensional NumPy arrays or tensors. For grayscale images, it's a 2D array (height x width) of pixel intensity values. For color images, it's a 3D array (height x width x color channels, e.g., RGB). They are rarely represented directly as DataFrames in deep learning.
