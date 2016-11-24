package FeedForwardNeuralNetwork;

import java.io.Serializable;
import java.util.Random;


public class RandomWrapper implements Serializable
{
	private Random rand;
	
	public RandomWrapper()
	{
		rand = new Random();
	}
	
	public void recreate()
	{
		rand = new Random();
	}

	
	public Random getRand()
	{
		return rand;
	}

}
