package t4.NeuralNet;

import t4.NeuralNet.Activation.ActivationFunction;

/**
 * A Bridge represents a link between a Neuron and one of its
 * input Neurons, and the corresponding weight between them.
 * 
 * @author Teddy
 */
public class Bridge {
	protected Neuron neuron;
	protected double weight;
	
	/**
	 * Constructs a new Bridge from the given Neuron with the
	 * specified weight.
	 * @param neuron Neuron used as the input for this Bridge
	 * @param weight Weight for the Neuron's input
	 */
	protected Bridge(Neuron neuron, double weight) {
		this.neuron = neuron;
		this.weight = weight;
	}
	
	/**
	 * Calculates the weighted input of the Neuron for this Bridge.
	 * @return Input Neuron's value multiplied by the weight of
	 * the Bridge
	 */
	protected double getWeightedInput(ActivationFunction func) {
		return neuron.getOutput(func) * this.weight;
	}
}
