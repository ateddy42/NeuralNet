package t4.NeuralNet;

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
	protected BridgeInput input;
	
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
	 * Constructs a new Bridge starting from the given Neuron
	 * with the specified weight.
	 * @param neuron Neuron used as the input for this Bridge
	 * @param weight Weight for the Neuron's input
	 */
	protected Bridge(BridgeInput input, Neuron output, double weight) {
		this.input = input;
		this.output = output;
		this.weight = weight;
		output.inputs.add(this);
	}
	
	/**
	 * Calculates the weighted input of the Neuron for this Bridge.
	 * @return Input Neuron's value multiplied by the weight of
	 * the Bridge
	 */
	protected double getWeightedInput() {
		return input.getValue() * weight;
	}
	
	public String toString() {
		return String.valueOf(getWeightedInput());
	}
}
