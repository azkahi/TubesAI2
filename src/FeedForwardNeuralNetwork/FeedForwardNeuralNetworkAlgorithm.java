/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FeedForwardNeuralNetwork;

import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author user-ari
 */
public class FeedForwardNeuralNetworkAlgorithm {
    public final static double UPPER_THRESHOLD = +45.0;
    public final static double LOWER_THRESHOLD = -45.0;
    public final static double MAX = +1.0;
    public final static double MIN =  0.0;
    /**
     * Random number generator
     */
    protected RandomWrapper rand;
    protected Neuron[][] neurons;
    

    public FeedForwardNeuralNetworkAlgorithm(RandomWrapper aRand)
	{
		rand = aRand;
	}


protected double handlingTransfer(double activation)
    {
        double output = 0.0;

        if (activation < LOWER_THRESHOLD)
        {
            output = MIN;
        }
        else if (activation > UPPER_THRESHOLD)
        {
            output = MAX;
        }
        else
        {
            //Transfer
            output = 1.0 / (1.0 + Math.exp(-activation));;
        }

       return output;
    }


    private int getNumOutputNeurons() {
        if(neurons==null)
        {
            return 0;
        }

        return neurons[neurons.length-1].length;
    }


}
