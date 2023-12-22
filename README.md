# Recurrent-Neural-Networks
The main objective of this repositori is to implement an RNN from scratch and provide an easy explanation as well to make it useful for the readers. Implementing any neural network from scratch at least once is a valuable exercise. It helps you gain an understanding of how neural networks work and here we are implementing an RNN which has its own complexity and thus provides us with a good opportunity to hone our skills.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/0*c_I0vMTeN0augGW-.jpg">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/0*c_I0vMTeN0augGW-.jpg">
  <img alt="Neural-Network" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*c_I0vMTeN0augGW-.jpg">
</picture>

There are various tutorials that provide a very detailed information of the internals of an RNN. You can find some of the very useful references at the end of this post. I could understand the working of an RNN rather quickly but what troubled me most was going through the BPTT calculations and its implementation. I had to spent some time to understand and finally put it all together. Without wasting any more time, let us quickly go through the basics of an RNN first.

## What is an RNN?
A recurrent neural network is a neural network that is specialized for processing a sequence of data x(t)= x(1), . . . , x(τ) with the time step index t ranging from 1 to τ. For tasks that involve sequential inputs, such as speech and language, it is often better to use RNNs. In a NLP problem, if you want to predict the next word in a sentence it is important to know the words before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far.

**Architecture :** Let us briefly go through a basic RNN network.
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*JOkrQoJ3J3-451GzRcayRg.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*JOkrQoJ3J3-451GzRcayRg.png">
  <img alt="Simple Architecture" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*JOkrQoJ3J3-451GzRcayRg.png">
</picture>
The left side of the above diagram shows a notation of an RNN and on the right side an RNN being unrolled (or unfolded) into a full network. By unrolling we mean that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 3 words, the network would be unrolled into a 3-layer neural network, one layer for each word.

**Input :** x(t)​ is taken as the input to the network at time step t. For example, x1,could be a one-hot vector corresponding to a word of a sentence.

**Hidden state :** h(t)​ represents a hidden state at time t and acts as “memory” of the network. h(t)​ is calculated based on the current input and the previous time step’s hidden state: h(t)​ = f(U x(t)​ + W h(t−1)​). The function f is taken to be a non-linear transformation such as tanh, ReLU.

**Weights :** The RNN has input to hidden connections parameterized by a weight matrix U, hidden-to-hidden recurrent connections parameterized by a weight matrix W, and hidden-to-output connections parameterized by a weight matrix V and all these weights (U,V,W) are shared across time.

**Output :** o(t)​ illustrates the output of the network. In the figure I just put an arrow after o(t) which is also often subjected to non-linearity, especially when the network contains further layers downstream.

## Forward Pass
The ﬁgure does not specify the choice of activation function for the hidden units. Before we proceed we make few assumptions: 1) we assume the hyperbolic tangent activation function for hidden layer. 2) We assume that the output is discrete, as if the RNN is used to predict words or characters. A natural way to represent discrete variables is to regard the output o as giving the un-normalized log probabilities of each possible value of the discrete variable. We can then apply the softmax operation as a post-processing step to obtain a vector ŷ of normalized probabilities over the output.

The RNN forward pass can thus be represented by below set of equations.
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:558/format:webp/1*55c3opV_tqm3wUwcj0m-jg.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:558/format:webp/1*55c3opV_tqm3wUwcj0m-jg.png">
  <img alt="RNN forward pass equations" src="https://miro.medium.com/v2/resize:fit:558/format:webp/1*55c3opV_tqm3wUwcj0m-jg.png">
</picture>

This is an example of a recurrent network that maps an input sequence to an output sequence of the same length. The total loss for a given sequence of x values paired with a sequence of y values would then be just the sum of the losses over all the time steps. We assume that the outputs o(t) are used as the argument to the softmax function to obtain the vector ŷ of probabilities over the output. We also assume that the loss L is the negative log-likelihood of the true target y(t) given the input so far.

## Backward Pass
The gradient computation involves performing a forward propagation pass moving left to right through the graph shown above followed by a backward propagation pass moving right to left through the graph. The runtime is O(τ) and cannot be reduced by parallelization because the forward propagation graph is inherently sequential; each time step may be computed only after the previous one. States computed in the forward pass must be stored until they are reused during the backward pass, so the memory cost is also O(τ). The back-propagation algorithm applied to the unrolled graph with O(τ) cost is called back-propagation through time (BPTT). Because the parameters are shared by all time steps in the network, the gradient at each output depends not only on the calculations of the current time step, but also the previous time steps.

## Computing Gradients
Given our loss function L, we need to calculate the gradients for our three weight matrices U, V, W, and bias terms b, c and update them with a learning rate α. Similar to normal back-propagation, the gradient gives us a sense of how the loss is changing with respect to each weight parameter. We update the weights W to minimize loss with the following equation:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:352/format:webp/1*S7N55YQK2xgH-jGWT0qL-w.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:352/format:webp/1*S7N55YQK2xgH-jGWT0qL-w.png">
  <img alt="equation to minimize loss" src="https://miro.medium.com/v2/resize:fit:352/format:webp/1*S7N55YQK2xgH-jGWT0qL-w.png">
</picture>
The same is to be done for the other weights U, V, b, c as well.

Let us now compute the gradients by BPTT for the RNN equations above. The nodes of our computational graph include the parameters U, V, W, b and c as well as the sequence of nodes indexed by t for x (t), h(t), o(t) and L(t). For each node n we need to compute the gradient ∇nL recursively, based on the gradient computed at nodes that follow it in the graph.

**Gradient with respect to output o(t)** is calculated assuming the o(t) are used as the argument to the softmax function to obtain the vector ŷ of probabilities over the output. We also assume that the loss is the negative log-likelihood of the true target y(t).
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*geEKJSbo6iOciOREkwuTuA.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*geEKJSbo6iOciOREkwuTuA.png">
  <img alt="Gradient with respect to output" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*geEKJSbo6iOciOREkwuTuA.png">
</picture>

Let us now understand how the gradient flows through hidden state h(t). This we can clearly see from the below diagram that at time t, hidden state h(t) has gradient flowing from both current output and the next hidden state.
Red arrow shows gradient flow
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*jcFuOZ3CiBHcmaxJ1wNMkw.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*jcFuOZ3CiBHcmaxJ1wNMkw.png">
  <img alt="Red arrow shows gradient flow" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*jcFuOZ3CiBHcmaxJ1wNMkw.png">
</picture>

We work our way backward, starting from the end of the sequence. At the ﬁnal time step τ, h(τ) only has o(τ) as a descendant, so its gradient is simple:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:568/format:webp/1*RTD0T0IP_uzGNLp9NpjEJA.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:568/format:webp/1*RTD0T0IP_uzGNLp9NpjEJA.png">
  <img alt="gradient" src="https://miro.medium.com/v2/resize:fit:568/format:webp/1*RTD0T0IP_uzGNLp9NpjEJA.png">
</picture>
We can then iterate backward in time to back-propagate gradients through time, from t=τ −1 down to t = 1, noting that h(t) (for τ > t ) has as descendants both o(t) and h(t+1). Its gradient is thus given by:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:786/format:webp/1*cmWW776Kc518CzYzifcZqQ.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:786/format:webp/1*cmWW776Kc518CzYzifcZqQ.png">
  <img alt="gradient" src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*cmWW776Kc518CzYzifcZqQ.png">
</picture>
Once the gradients on the internal nodes of the computational graph are obtained, we can obtain the gradients on the parameter nodes. The gradient calculations using the chain rule for all parameters is:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:786/format:webp/1*5O4iMbB_EklXw76pwVZZsw.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:786/format:webp/1*5O4iMbB_EklXw76pwVZZsw.png">
  <img alt="gradient" src="https://miro.medium.com/v2/resize:fit:786/format:webp/1*5O4iMbB_EklXw76pwVZZsw.png">
</picture>
We are not interested to derive these equations here, rather implementing these. There are very good posts here and here providing detailed derivation of these equations.

## Implementation
We will implement a full Recurrent Neural Network from scratch using Python. We will try to build a text generation model using an RNN. We train our model to predict the probability of a character given the preceding characters. It’s a generative model. Given an existing sequence of characters we sample a next character from the predicted probabilities, and repeat the process until we have a full sentence. This implementation is from Andrej Karparthy great post building a character level RNN. Here we will discuss the implementation details step by step.

**General steps to follow:**
* Initialize weight matrices U, V, W from random distribution and bias b, c with zeros
* Forward propagation to compute predictions
* Compute the loss
* Back-propagation to compute gradients
* Update weights based on gradients
* Repeat steps 2–5

Let's see the implementation now in Python.
