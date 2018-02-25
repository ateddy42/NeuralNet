package t4.NeuralNet.Activation;

/**
 * A Sigmoid ActivationFunction is a function that takes
 * values and returns a value in the range [0, 1]. An input
 * of 0 returns 0.5, larger input values return larger
 * output values, and smaller input values return smaller.
 * 
 * @author Teddy
 */
public class Sigmoid extends ActivationFunction {
	
	/**
	 * {@inheritDoc}
	 * 
	 * This implementation uses the Sigmoid function:
	 * <br><br><code>1 / (1 + Math.exp(-input))</code>
	 */
	public double getOutput(double input) {
		return 1 / (1 + Math.exp(-input));
	}
	
	/**
	 * {@inheritDoc}
	 * 
	 * This implementation uses the Sigmoid function:
	 * <br><br><code>f = 1 / (1 + Math.exp(-input))</code>
	 * <br><code>f' = f * (1 - f)</code>
	 */
	public double getDerivative(double f) {
		return f * (1 - f);
	}

}
