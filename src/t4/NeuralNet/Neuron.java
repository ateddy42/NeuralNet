package t4.NeuralNet;

import t4.NeuralNet.Activation.ActivationFunction;

/**
 * A Neuron represents one node in a NeuralNet. It has
 * a value and a collection of Bridge inputs, linking
 * it to other Neurons in previous layers.
 * 
 * @author Teddy
 */
public class Neuron {
	/**
	 * Output value of this neuron
	 */
	protected double value;
	/**
	 * Layer this Neuron is for
	 */
	protected Layer layer;
	/**
	 * Set of Bridges marking connections to Neurons in the previous layer
	 */
	protected Bridge[] inputs;
	
	/**
	 * Constructs a new Neuron with the given value, number
	 * of input Bridges, and the Layer it resides in.
	 * @param value Value of the Neuron
	 * @param numInputs Number of input Bridges
	 * @param layer Layer this Neuron is part of
	 */
	protected Neuron(double value, int numInputs, Layer layer) {
		this.value = value;
		this.inputs = new Bridge[numInputs];
		this.layer = layer;
	}
	
	/**
	 * Update the value of the Neuron to be the weighted sum
	 * of the Neurons linked via the input Bridges.
	 * @param func Activation Function
	 * @return Updated value for this Neuron
	 */
	protected double calculateValue(ActivationFunction func) {
		if (inputs.length != 0) {
			value = 0;
			for (int i = 0; i < inputs.length; i++) {
				value += inputs[i].getWeightedInput(func);
			}
			value = func.getOutput(value);
		}
		return this.value;
	}
	
	/**
	 * Computes the output from this Neuron using the given
	 * activation function
	 * @param func Activation function
	 * @return Output from this Neuron
	 */
	public double getOutput(ActivationFunction func) {
		if (inputs.length == 0)
			return value;
		return func.getOutput(value);
	}
	
	/**
	 * Sets the value of this Neuron to the given value
	 * @param value New value for this Neuron
	 */
	public void setValue(double value) {
		this.value = value;
	}
	
	/**
	 * Update the values of the weights for the input Bridges connected
	 * to this Neuron using the following function:<br><br>
	 * <code>w = w + alpha * (d - f) * a' * x</code><br><br>
	 * Where:
	 * <br>-w is the current weight
	 * <br>-alpha is the learning rate
	 * <br>-d is the desired output
	 * <br>-f is the observed output
	 * <br>-a' is the derivative of the activation function
	 * <br>-x is the input value to the bridge's weight
	 * 
	 * @param alpha Learning rate
	 * @param f Observed output
	 * @param d Desired output
	 * @param payoff Payoff of this move
	 * @param func Activation funciton
	 */
	protected void backpropagate(double alpha, double f, double d,
			double payoff, ActivationFunction func) {
		double delta = payoff * alpha * (d - f) * func.getDerivative(f);
		for (Bridge b : inputs) {
			b.weight += delta * b.neuron.getOutput(func);
		}
	}
	
	public String toString() {
		return String.valueOf(value);
	}
}
