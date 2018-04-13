package t4.NeuralNet;

public class BridgeInput {
	
	protected double value;

	protected BridgeInput(double value) {
		this.value = value;
	}
	
	protected double getValue() {
		return value;
	}
	
	protected void setValue(double value) {
		this.value = value;
	}
	
	public String toString() {
		return String.valueOf(value);
	}

}
