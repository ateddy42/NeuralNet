### NeuralNet

Experimantal Java implementation of a Neural Network. Contains framework for a NeuralNet consisting of any number of Layers, each with a different number of Neurons, connected to previous layers by Bridges.

Sample implementation:

```
// Create NeuralNet
nn = new NeuralNet(new Sigmoid());

// Create Input Layer
new Layer(nn, NUM_INPUTS, "Input Layer");

// Create Hidden Layer(s)
for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
    new Layer(nn, NUM_HIDDEN_NEURONS, "Hidden Layer " + i);
}

// Create Output Layer
new Layer(nn, NUM_OUTPUTS, "Output Layer");
```