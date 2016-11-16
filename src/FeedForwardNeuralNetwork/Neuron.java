/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FeedForwardNeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;

/**
 *
 * @author user-ari
 */
public class Neuron  {
    
    public final static double DEFAULT_BIAS_VALUE = 1.0;
    
    protected double biasInputValue;

    // weights to apply to inputs (num inputs + 1 for the bias)
    protected List<Double> inputWeights;

    // derivatives of error in regard to weights
    protected List<Double> error;

    // last change in each weight
    protected List<Double> lastWeightDeltas;

    // index in the weight vector of the bias weight (always at the end of the array)
    protected int biasIndex;
    
    public Neuron(int number , double input_bias){
        biasInputValue = input_bias;
        
        inputWeights = new ArrayList<Double>();
        error = new ArrayList<Double>();
        lastWeightDeltas = new ArrayList<Double>();
        
        //biasIndex = number;
        
    }
    
    public double preactivation(Instance instance){
        double result = 0.0;
        double [] input = instance.toDoubleArray();
        int offset = 0;

        for(int i=0; i<input.length; i++)
        {
            // class values are not included
            if(i != instance.classIndex())
            {
                // never add missing values into the activation
                if(instance.isMissing(i))
                {
                    offset++;
                }
                else
                {
                    result += (input[i] * inputWeights.get(offset++));
                }
            }
        }

        // add the bias output
        result += (biasInputValue * inputWeights.get(biasIndex).doubleValue());

        return result;
    }
    
    public double preactivation(double [] inputs)
    {
        // calculate the activation given an input vector

        double result = 0.0;

        for(int i=0; i<inputs.length; i++)
        {
            result += (inputs[i] * inputWeights.get(i).doubleValue());
        }

        // add the bias output
        result += (biasInputValue * inputWeights.get(biasIndex).doubleValue());

        return result;
    }
    
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
    
    
    
    public double activate (Instance instance){
        return sigmoid(preactivation(instance));
    }
    
    public double activate (double[] inputs){
        return sigmoid(preactivation(inputs));
    }
    
    
    public List<Double> getError(){
        return error;
    }
    
    public List<Double> getLastWeightDeltas()
    {
        return lastWeightDeltas;
    }

    public List<Double> getWeights()
    {
        return inputWeights;
    }

    public int getBiasIndex()
    {
        return biasIndex;
    }
    
    /**
     * @return
     */
    public double getBiasInputValue()
    {
    	return biasInputValue;
    }

}
