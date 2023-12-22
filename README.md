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

















































































