package t4.NeuralNet;

/**
 * A Layer represents a collection of Neurons in one of
 * the NeuralNet's various layers.
 * 
 * @author Teddy
 */
public class Layer {
	/**
	 * Array of neurons in this layer
	 */
	protected Neuron[] neurons;
	
	/**
	 * NeuralNet this layer is part of
	 */
	protected NeuralNet nn;
	
	/**
	 * Previous layer
	 */
	protected Layer previous;
	
	/**
	 * Name of this layer
	 */
	private String name;
	
	/**
	 * Constructs a new Layer with a given number of Neurons.
	 * Initializes the Neurons with default values (see {@link NeuralNet})
	 * and connects them with Bridges to the neurons in the previous layer
	 * as well as the NeuralNet bias.
	 * @param nn NeuralNet this layer should be added to
	 * @param numNeurons Number of Neurons in this Layer
	 * @param name Name of this layer
	 */
	public Layer(NeuralNet nn, int numNeurons, String name) {
		this.nn = nn;
		this.name = name;
		this.neurons = new Neuron[numNeurons];
		previous = nn.getOutputLayer();
		int numInputs = previous == null ? 0 : previous.neurons.length + 1;
		for (int i = 0; i < numNeurons; i++) {
			Neuron neuron = new Neuron(0, numInputs, this);
			// check if previous layer exists
			if (previous != null) {
				// add bridge to bias
				neuron.inputs[0] = new Bridge(nn.bias,
						NeuralNet.INIT_BIAS_WEIGHT);
				
				// add bridge to all previous layer neurons
				for (int j = 1; j < numInputs; j++) {
					neuron.inputs[j] = new Bridge(
							previous.neurons[j - 1],
							NeuralNet.INIT_NEURON_WEIGHT);
				}
			}
			neurons[i] = neuron;
		}
		nn.layers.add(this);
	}
	
	/**
	 * Checks whether this Layer has a previous Layer, or if
	 * it is the first Layer
	 * @return Whether this Layer has a previous Layer
	 */
	protected boolean isInputLayer() {
		return previous == null;
	}
	
	/**
	 * Update the values for each of the Neurons in this Layer
	 * as the weighted sum of the input to each Neuron. If a
	 * Neuron does not have input values, its value is set to 0.
	 */
	protected void updateValues() {
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].calculateValue(nn.activation);
		}
	}
	
	/**
	 * Return an array of values for this Layer, corresponding to
	 * each of the Neuron's values.
	 * @return Array of values for this Layer's Neurons
	 */
	protected double[] getValues() {
		double[] values = new double[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			values[i] = neurons[i].value;
		}
		return values;
	}
	
	/**
	 * Sets the values for each of the Neurons in this Layer
	 * @param values Array of values
	 * @throws IndexOutOfBoundsException If number of values != number of Neurons
	 */
	protected void setValues(double[] values) throws IndexOutOfBoundsException {
		if (neurons.length != values.length)
			throw new IndexOutOfBoundsException("Number of input values does not match the number of Neurons");
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].setValue(values[i]);
		}
	}
	
	public String toString() {
		return this.name;
	}
}
