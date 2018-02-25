package t4.NeuralNet.Activation;

/**
 * A Threshold ActivationFunction is a simple function that
 * returns 1 if the input is above a specific value, otherwise
 * it returns 0.
 * 
 * @author Teddy
 */
public class Threshold extends ActivationFunction {
	public static double THRESHOLD = 0;
	
	/**
	 * {@inheritDoc}
	 * 
	 * This implementation returns 1 if the input value is
	 * greate than the THRESHOLD value, else 0.
	 */
	public double getOutput(double input) {
		return  (input > THRESHOLD) ? 1 : 0;
	}

	/**
	 * {@inheritDoc}
	 * 
	 * This implementation returns a constant of 1.
	 */
	public double getDerivative(double f) {
		return 1;
	}
}
