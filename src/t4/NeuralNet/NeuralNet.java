package t4.NeuralNet;

import java.util.ArrayList;

import t4.NeuralNet.Activation.ActivationFunction;
import t4.NeuralNet.Activation.Threshold;

/**
 * A NeuralNet is a representation of a Neural Network.
 * It has a Collection of Layers containing Neurons linked
 * together by Bridges. A NeuralNet also has a bias Neuron,
 * containing the bias used for all Neurons outside the
 * input layer, a learning rate alpha, and an activation
 * function used by each of the Neurons.
 * 
 * @author Teddy
 */
public class NeuralNet {
	/**
	 * Default learning rate
	 */
	public static final double ALPHA = 0.5;
	/**
	 * Default input bias
	 */
	public static final double BIAS = 1;
	/**
	 * Initial weight for bias value
	 */
	public static double INIT_BIAS_WEIGHT = -1;
	/**
	 * Initial weight for neuron value
	 */
	public static double INIT_NEURON_WEIGHT = 1;
	
	/**
	 * Layers of this NeuralNet
	 */
	protected ArrayList<Layer> layers;
	/**
	 * Input Bias Neuron
	 */
	protected Neuron bias;
	/**
	 * Learning rate of this NeuralNet
	 */
	protected double alpha;
	/**
	 * Activation Function for this NeuralNet
	 */
	protected ActivationFunction activation;
	
	/**
	 * Constructs a new NeuralNet object with the Threshold activation
	 * function, and the default learning rate and bias value.
	 */
	public NeuralNet() {
		this(new Threshold(), ALPHA, BIAS);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function. Uses the default learning rate and bias values.
	 * @param activation Activation Function
	 */
	public NeuralNet(ActivationFunction activation) {
		this(activation, ALPHA, BIAS);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function and learning rate. Uses the default bias value.
	 * @param activation Activation function
	 * @param alpha Learning rate
	 */
	public NeuralNet(ActivationFunction activation, double alpha) {
		this(activation, alpha, BIAS);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function, learning rate, and bias value.
	 * @param activation Activation function
	 * @param alpha Learning rate
	 * @param bias Value for the bias
	 */
	public NeuralNet(ActivationFunction activation, double alpha, double bias) {
		layers = new ArrayList<>();
		this.activation = activation;
		this.alpha = alpha;
		this.bias = new Neuron(bias, null);
	}
	
	/**
	 * Return the final layer of this NeuralNet
	 * @return Output Layer
	 */
	protected Layer getOutputLayer() {
		if (layers.isEmpty()) return null;
		return layers.get(layers.size() - 1);
	}
	
	/**
	 * Return the first layer of this NeuralNet
	 * @return Input Layer
	 */
	protected Layer getInputLayer() {
		if (layers.isEmpty()) return null;
		return layers.get(0);
	}
	
	/**
	 * Sets the values for the input layer from the given array
	 * @param values Input values for the NeuralNet
	 * @throws IndexOutOfBoundsException If number of values != number of Neurons
	 */
	public void setInputValues(double[] values) throws IndexOutOfBoundsException {
		getInputLayer().setValues(values);
	}
	
	/**
	 * Updates the calculated values for each layer, and returns the
	 * values for the output layer
	 * @return Values for the output layer
	 */
	public double[] getOutputValues() {
		// update values for all layers, in order
		for (Layer layer : layers) {
			layer.updateValues();
		}
		// fetch updated values for output layer
		return getOutputLayer().getValues();
	}
	
	/**
	 * Update the values of the weights for all Bridges in this
	 * NeuralNet for the non-null entries in the <code>desired</code>
	 * array.
	 * @param input Values for the input layer
	 * @param desired Desired output values
	 * @throws IndexOutOfBoundsException If number of values != number of Neurons
	 */
	public void backpropagate(double[] input, double[] desired) throws IndexOutOfBoundsException {
		backpropagate(input, desired, 1);
	}
	
	/**
	 * Update the values of the weights for all Bridges in this
	 * NeuralNet for the non-null entries in the <code>desired</code>
	 * array.
	 * @param input Values for the input layer
	 * @param desired Desired output values
	 * @param payoff Payoff for this set of inputs
	 * @throws IndexOutOfBoundsException If number of values != number of Neurons
	 */
	public void backpropagate(double[] input, double[] desired,
			double payoff) throws IndexOutOfBoundsException {
		setInputValues(input);
		Layer layer = this.getOutputLayer();
		layer.setDesired(desired);
		while (layer != null) {
			layer.backpropagate(payoff);
			layer = layer.previous;
		}
	}
}
