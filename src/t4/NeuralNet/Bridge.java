package t4.NeuralNet;

import t4.NeuralNet.Activation.ActivationFunction;

/**
 * A Bridge represents a link between a Neuron and one of its
 * input Neurons, and the corresponding weight between them.
 * 
 * @author Teddy
 */
public class Bridge {
	/**
	 * Input Neuron
	 */
	protected Neuron input;
	
	/**
	 * Output Neuron
	 */
	protected Neuron output;
	
	/**
	 * Weight associated with this Bridge
	 */
	protected double weight;
	
	/**
	 * Constructs a new Bridge starting from the given Neuron
	 * with the specified weight.
	 * @param neuron Neuron used as the input for this Bridge
	 * @param weight Weight for the Neuron's input
	 */
	protected Bridge(Neuron input, Neuron output, double weight) {
		this.input = input;
		this.output = output;
		this.weight = weight;
		input.outputs.add(this);
		output.inputs.add(this);
	}
	
	/**
	 * Calculates the weighted input of the Neuron for this Bridge.
	 * @param func Activation Function
	 * @return Input Neuron's value multiplied by the weight of
	 * the Bridge
	 */
	protected double getWeightedInput(ActivationFunction func) {
		return input.getOutput(func) * this.weight;
	}
	
	public String toString() {
		return String.valueOf(weight);
	}
}
