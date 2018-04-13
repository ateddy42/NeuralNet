package t4.NeuralNet;

import java.util.ArrayList;

import t4.NeuralNet.Activation.ActivationFunction;

/**
 * A Neuron represents one node in a NeuralNet. It has
 * a value and a collection of Bridge inputs, linking
 * it to other Neurons in previous layers.
 * 
 * @author Teddy
 */
public class Neuron extends BridgeInput {
	/**
	 * Desired output value of this neuron
	 */
	protected Double desired;
	/**
	 * Layer this Neuron is for
	 */
	protected Layer layer;
	/**
	 * Set of Bridges marking connections to Neurons in the previous layer
	 */
	protected ArrayList<Bridge> inputs;
	/**
	 * Set of Bridges marking connections to Neurons in the next layer
	 */
	protected ArrayList<Bridge> outputs;
	
	/**
	 * Constructs a new Neuron with the given value, number
	 * of input Bridges, and the Layer it resides in.
	 * @param value Value of the Neuron
	 * @param layer Layer this Neuron is part of
	 */
	protected Neuron(double value, Layer layer) {
		super(value);
		this.inputs = new ArrayList<>();
		this.outputs = new ArrayList<>();
		this.layer = layer;
	}
	
	protected void link(Neuron n, double weight) {
		new Bridge(this, n, weight);
	}
	
	/**
	 * Update the value of the Neuron to be the weighted sum
	 * of the Neurons linked via the input Bridges.
	 * @param func Activation Function
	 * @return Updated value for this Neuron
	 */
	protected double calculateValue(ActivationFunction func) {
		// update value based on weighted inputs
		value = 0.0;
		for (Bridge b : inputs) {
			value += b.getWeightedInput();
		}
		value = func.getOutput(value);
		
		return this.value;
	}
	
	/**
	 * Sets the desired output of this Neuron to the given value
	 * @param value New desired output for this Neuron
	 */
	protected void setDesired(double desired) {
		this.desired = desired;
	}
	
	/**
	 * Update the values of the weights for the input Bridges connected
	 * to this Neuron using the following function:<br><br>
	 * <code>w = w + payoff * alpha * delta * x</code><br><br>
	 * Where:
	 * <br>-w is the current weight
	 * <br>-payoff is the payoff of this move
	 * <br>-alpha is the learning rate
	 * <br>-delta is the calculated delta for this neuron
	 * <br>-x is the input value to the bridge's weight
	 * 
	 * @param alpha Learning rate
	 * @param f Observed output
	 * @param payoff Payoff of this move
	 * @param func Activation function
	 */
	protected void backpropagate(double alpha, double payoff,
			ActivationFunction func) {
		double change = payoff * alpha * calculateDelta(func);
		for (Bridge b : inputs) {
			b.weight += change * b.input.getValue();
		}
	}
	
	/**
	 * Calculate the delta value for this Neuron. For the output layer,
	 * this returns the real output times the gradient:<br><br>
	 * <code>delta_i = (desired_i - value_i) * gradient</code><br><br>
	 * For layer l which is not an output layer, this returns the gradient
	 * times the sum of the weighted deltas for the l+1 layer:<br><br>
	 * <code>delta_i = gradient * Sum(delta_j * weight_i_j)</code><br><br>
	 * Where:
	 * <br>-j is a neuron in the l+1 layer
	 * <br>-weight_i_j is the weight of the Bridge from Neuron i to j
	 * @param func Activation function
	 * @return
	 */
	private double calculateDelta(ActivationFunction func) {
		if (desired != null) {
			// output layer, so return real output error times the gradient
			return (desired - value) * func.getDerivative(value);
		}
		double sumOfOutputDeltas = 0;
		for (Bridge b : outputs) {
			sumOfOutputDeltas += b.output.calculateDelta(func) * b.weight;
		}
		return func.getDerivative(calculateValue(func)) * sumOfOutputDeltas;
	}
	
	public String toString() {
		return String.valueOf(value);
	}
}
