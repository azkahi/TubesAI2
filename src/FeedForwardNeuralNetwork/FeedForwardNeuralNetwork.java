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
    private FeedForwardNeuralNetworkAlgorithm FFNN;
    
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
        FFNN = new FeedForwardNeuralNetworkAlgorithm(instances);
        FFNN.buildModel(hidden_layer, hidden_neurons);
        
        FFNN.printModel();
        FFNN.printAllWeights();
        
        //Start Training
        double[] input = null;
        double error = 0;
        int correct = 0;
        int incorrect =0;
        int j = 1;
        //j itu buat ngatur banyaknya iterasi training
        //makin banyak makin lama tapi makin akurat
        while (FFNN.getSumError() != 0 && j < 1000){
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
    
    //Ini buat nampilin output array
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        FFNN.setInputLayer(instance.toDoubleArray());
        FFNN.determineOutput(instance);
        FFNN.updateModel(instance);
        double[] result = new double[FFNN.getNeurons().length];
        for (int i = 0 ; i < result.length ; i++){
            result[i] = (FFNN.getNeurons())[FFNN.getNeurons().length-1][i].getOutputValue();
        }
        
        return result;
        
                
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
         double[] array = distributionForInstance(instance);
         double max = array[0];
         int idx = 0;
         for (int i = 1 ; i < array.length ; i++){
             if (array[i] > max){
                 max = array[i];
                 idx = i;
             }
         }
         
         return array[idx];
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