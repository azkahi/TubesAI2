/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FeedForwardNeuralNetwork;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;



/**
 *
 * @author user-ari
 */
public class FeedForwardNeuralNetwork extends AbstractClassifier
{

    @Override
    public void buildClassifier(Instances i) throws Exception {
        
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
        int j = 1;
        while (FFNN.getSumError() != 0 && j < 1000 && (FFNN.getClassOutputValues()!= 0 || FFNN.getClassOutputValues()!= -1)){
            System.out.println("Iterasi ke - "+j);
            for (int i = 0; i<instances.size(); i++){
                error = 0;
                FFNN.clearModel();
                input = instances.get(i).toDoubleArray();
                FFNN.setInputLayer(input);
                FFNN.determineOutput(instances.get(i));
                FFNN.updateModel();
                System.out.println("DATA KE "+(i+1));
                FFNN.printModel();
                FFNN.printAllWeights();
                System.out.println("Class : "+FFNN.getClassOutputValues());
                System.out.println("Error : "+FFNN.getSumError());
            }
            j++;
        }
        
        
        
        
    }
    

}