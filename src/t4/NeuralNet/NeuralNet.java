package t4.NeuralNet;

import java.util.ArrayList;
import java.util.Random;

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
	 * Initial value for simmulated annealing
	 */
	public static final double INIT_BETA = 0.75;
	/**
	 * Rate of decay for {@link beta}. Values must be
	 * between 0 and 1, exclusive. Higher values decay slower.
	 */
	public static final double BETA_DECAY_RATE = 0.95;
	/**
	 * Default input bias
	 */
	public static final double BIAS = 1;
	/**
	 * Initial weight for bias value
	 */
	public static double INIT_BIAS_WEIGHT = -1;
	/**
	 * Instance used to generate random values
	 */
	protected Random rand;
	/**
	 * Whether to use simulated annealing
	 */
	protected boolean useAnnealing;
	/**
	 * Number of rounds of backpropagation completed
	 */
	private int numRounds;
	/**
	 * Layers of this NeuralNet
	 */
	protected ArrayList<Layer> layers;
	/**
	 * Input values of this NeuralNet
	 */
	protected BridgeInput[] inputs;
	/**
	 * Input Bias
	 */
	protected BridgeInput bias;
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
	public NeuralNet(int numInputs) {
		this(numInputs, new Threshold(), ALPHA, BIAS, false);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function. Uses the default learning rate and bias values.
	 * @param activation Activation Function
	 */
	public NeuralNet(int numInputs, ActivationFunction activation) {
		this(numInputs, activation, ALPHA, BIAS, false);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function and learning rate. Uses the default bias value.
	 * @param activation Activation function
	 * @param alpha Learning rate
	 */
	public NeuralNet(int numInputs, ActivationFunction activation, double alpha) {
		this(numInputs, activation, alpha, BIAS, false);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function, learning rate, and bias value.
	 * @param activation Activation function
	 * @param alpha Learning rate
	 * @param bias Value for the bias
	 */
	public NeuralNet(int numInputs, ActivationFunction activation, double alpha, double bias) {
		this(numInputs, activation, alpha, bias, false);
	}
	
	/**
	 * Constructs a new NeuralNet object with the given activation
	 * function, learning rate, and bias value.
	 * @param activation Activation function
	 * @param alpha Learning rate
	 * @param bias Value for the bias
	 * @param useAnnealing Whether to use simmulated annealing
	 */
	public NeuralNet(int numInputs, ActivationFunction activation, double alpha,
			double bias, boolean useAnnealing) {
		rand = new Random();
		layers = new ArrayList<>();
		inputs = new BridgeInput[numInputs];
		for (int i = 0; i < numInputs; i++) {
			inputs[i] = new BridgeInput(0);
		}
		this.activation = activation;
		this.alpha = alpha;
		this.bias = new BridgeInput(bias);
		this.numRounds = 0;
		this.useAnnealing = useAnnealing;
	}
	
	/**
	 * Add a new layer to the Neural Network with the given name
	 * and number of neurons
	 * @param numNeurons Number of Neurons in the Layer
	 * @param name Name of the Layer
	 */
	public void addLayer(int numNeurons, String name) {
		new Layer(this, numNeurons, name);
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
	 * Returns 1 + the a random double between 0 and 1. This
	 * is used as the initial weight for new Bridges.
	 * @return Initial weight for new Bridges
	 */
	protected double getInitialBridgeWeight() {
		return 1 + rand.nextDouble();
	}
	
	/**
	 * Sets the values for the input layer from the given array
	 * @param inputs Input values for the NeuralNet
	 */
	public void setInputValues(double[] inputs) {
		for (int i = 0; i < inputs.length; i++)
			this.inputs[i].setValue(inputs[i]);
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
	public void backpropagate(double[] input, double[] desired)
			throws IndexOutOfBoundsException {
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
	public void backpropagate(double[] input, double[] desired, double payoff)
			throws IndexOutOfBoundsException {
		numRounds++;
		setInputValues(input);
		getOutputValues();
		Layer layer = this.getOutputLayer();
		layer.setDesired(desired);
		while (layer != null) {
			layer.backpropagate(payoff);
			layer = layer.previous;
		}
	}
	
	/**
	 * Return a random amount used for annealing. Calculated by
	 * taking the initial beta value and dividing by the number
	 * of rounds times the beta decay rate. If the random number
	 * is greater than this value, return 0, else return this
	 * random number.
	 * @return Amount used for simmulated annealing
	 */
	protected double getAnnealingAmount() {
		double d = rand.nextDouble();
		if (!useAnnealing || d > INIT_BETA / (numRounds * BETA_DECAY_RATE))
			return 0;
		return d;
	}
}
