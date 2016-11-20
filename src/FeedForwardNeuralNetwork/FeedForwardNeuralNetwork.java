/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FeedForwardNeuralNetwork;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;



/**
 *
 * @author user-ari
 */
public class FeedForwardNeuralNetwork extends AbstractClassifier
{

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);
        
        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        
        trainModel(instances,1,5);
    }
    
    
    public void trainModel(Instances instances, int hidden_layer, int hidden_neurons){
        //Initialize
        FeedForwardNeuralNetworkAlgorithm FFNN = new FeedForwardNeuralNetworkAlgorithm(instances);
        FFNN.buildModel(hidden_layer, hidden_neurons);
        
        FFNN.printModel();
        FFNN.printAllWeights();
        
        //Start Training
        double[] input = null;
        double error = 0;
        int correct = 0;
        int incorrect =0;
        int j = 1;
        while (FFNN.getSumError() != 0 && j < 100 /*&& (FFNN.getClassOutputValues()!= 0 || FFNN.getClassOutputValues()!= -1)*/){
            System.out.println("\n\n\nIterasi ke - "+j);
            for (int i = 0; i<instances.size(); i++){
                error = 0;
                FFNN.clearModel();
                input = instances.get(i).toDoubleArray();
                FFNN.setInputLayer(input);
                FFNN.determineOutput(instances.get(i));
                FFNN.updateModel(instances.get(i));
                System.out.println("\nIterasi ke - "+j+" DATA KE "+(i+1)+"\n");
                FFNN.printModel();
                FFNN.printAllWeights();
                System.out.println("Class : "+FFNN.getClassOutputValues());
                System.out.println("Error : "+FFNN.getSumError());
                if (FFNN.getClassOutputValues() == instances.get(i).classValue())
                    correct++;
                else
                    incorrect++;
            }
            j++;
        }
        System.out.println("Correct : "+correct);
        System.out.println("Incorrect : "+incorrect);
        
        
        
        
    }
    
    //INI GUA MASIH TEUING, TOLONG DIKAJI WKWK
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }
    
}