/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FeedForwardNeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
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
    
    protected Instances instances;
    protected Neuron[][] neurons;
    protected int hidden_layers;
    private final RandomWrapper rnd = new RandomWrapper();
    

    public void buildModel(int n_hidden_layer, int neuron_hidden_layer){
        hidden_layers = n_hidden_layer;
        //Ini weight, belum sesuai input
        List<Double> arr = new ArrayList<>();
        
        
        //count total number of neurons
        int num_neurons = 0;
        switch (n_hidden_layer) {
            case 0:
                num_neurons = (instances.numAttributes()-1)+ instances.numClasses();
                break;
            case 1:
                num_neurons = (instances.numAttributes()-1)+ instances.numClasses() + neuron_hidden_layer;
                break;
            default:
                throw new RuntimeException("Illegal n_hidden_layer");
        }
        //Build array container
        neurons = new Neuron[n_hidden_layer+2][];
        //Input Layer
        neurons[0] = new Neuron[instances.numAttributes()-1];
        for (int j=0; j< instances.numAttributes()-1; j++){
                neurons[0][j] = new Neuron();
        }
        
        //Hidden Layer
        neurons[1] = new Neuron[neuron_hidden_layer];
        if (n_hidden_layer == 1){
            for (int j=0; j< neuron_hidden_layer; j++){
                neurons[1][j] = new Neuron();
                for (int i=0; i<neurons[0].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[1][j].setWeights(arr);
                arr.clear();
            }
            //Output Layer
            neurons[2] = new Neuron[instances.numClasses()];
            for (int j=0; j< instances.numClasses(); j++){
                neurons[2][j] = new Neuron();
                for (int i=0; i<neurons[1].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[2][j].setWeights(arr);
                arr.clear();
            }
        }
        else //Hidden layer = 0
        {
            neurons[1] = new Neuron[instances.numClasses()];
            for (int j=0; j< instances.numClasses(); j++){
                neurons[1][j] = new Neuron();
                for (int i=0; i<neurons[0].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[1][j].setWeights(arr);
                arr.clear();
            }
        }
        
        
    }
    
    public void printModel(){
        for (int i = 0 ; i< neurons.length; i++){
            System.out.print("Layer "+i+" : ");
            for (int j =0 ; j < neurons[i].length ; j++){
                System.out.print(neurons[i][j].getValue()+" ");
            }
            System.out.println("");
        }
    }
    
    public FeedForwardNeuralNetworkAlgorithm(Instances i){
		instances = i;
                hidden_layers = -1;
    }
    
    public void setInputLayer(double[] inputs ){
        //Asumsi sudah sama panjangnya
        for (int i=0 ; i < neurons[0].length ; i++){
            neurons[0][i].setValue(inputs[i]);
        }
    } 
    
    public void updateModel(){
        double [] err = countOutput(instances.get(0));
        if (hidden_layers == 0){
            for (int i=0 ; i<neurons[1].length; i++){
                List<Double> weights = neurons[1][i].getWeights();
                double val = weights.get(i).doubleValue() + 
                weights.set(i, MIN);
            }
        }
        else if (hidden_layers == 1)
        {
            
        }
    }
    
    public double[] countOutputError(Instance instance){
        //Only invoke this after call countOutput method
        int classnum = neurons[neurons.length-1].length;
        double[] result = new double[classnum];
        for (int i=0; i<classnum; i++){
            double out = neurons[neurons.length-1][i].getValue();
            result[i] = out * (1-out) * (instance.classValue() - out);
        }
        return result;
    }
    
    public double[] countHiddenError(Instance instance){
        //only call this if there's hidden layer
        int hidnum = neurons[1].length;
        double[] result = new double[hidnum];
        
        //count sum of error * weight first
        int classnum = neurons[neurons.length-1].length;
        double sum = 0;
        double[] error = countOutputError(instance);
        for (int i=0; i<classnum; i++){
            sum += error[i] * neurons[1][i].getValue();
        }
        
        for (int i=0; i<hidnum; i++){
            double out = neurons[1][i].getValue();
            result[i] = out * (1-out) * (sum);
        }
        return result;
    }
    
    public double[] countOutput(Instance instance){
        setInputLayer(instance.toDoubleArray());
        
        double[] result = new double[instance.numClasses()];
        if (hidden_layers == 0){
            //Langsung hitung output
            for (int k = 0 ; k<instance.numClasses() ; k++){
                neurons[1][k].setValue(neurons[1][k].activate(instance));
            }
            Neuron max = neurons[1][0]; //Initialize
            //Cari nilai maksimal, karena kelas hanya bisa satu, sehingga kelas lain 
            //nilainya 0, sementara kelas maksimal diberi nilai 1
            for (int k = 1; k < instance.numClasses(); k++){
                if (max.getValue() < neurons[1][k].getValue()){
                    max = neurons[1][k];
                }
            }
            for (int k = 0; k < instance.numClasses(); k++){
                if (!neurons[1][k].equals(max)){
                    neurons[1][k].setValue(0);
                } else {
                    neurons[1][k].setValue(1);
                }
                result[k] = neurons[1][k].getValue();
            }
        }
        else if (hidden_layers == 1){
            //Hitung hidden layer
            for (int i=0 ; i<neurons[1].length ; i++){
                neurons[1][i].setValue(neurons[1][i].activate(instance));
                System.out.println("value 1 "+i+" : "+neurons[1][i].getValue());

            }
            //Hitung output layer
            for (int k = 0 ; k<instance.numClasses() ; k++){                 
                double[] inp = new double[neurons[1].length];
                for (int i=0; i < neurons[1].length ; i++){
                    inp[i] = neurons[1][i].getValue();
                }
                
                neurons[2][k].setValue(neurons[2][k].activate(inp));
                System.out.println("value 2 "+k+" : "+neurons[2][k].getValue());
            }
            
            Neuron max = neurons[2][0]; //Initialize
            //Cari nilai maksimal, karena kelas hanya bisa satu, sehingga kelas lain 
            //nilainya 0, sementara kelas maksimal diberi nilai 1
            for (int k = 1; k < instance.numClasses(); k++){
                if (max.getValue() < neurons[1][k].getValue()){
                    max = neurons[2][k];
                }
            }
            for (int k = 0; k < instance.numClasses(); k++){
                if (!neurons[2][k].equals(max)){
                    neurons[2][k].setValue(0);
                } else {
                    neurons[2][k].setValue(1);
                }
                result[k] = neurons[2][k].getValue();
            }
        }
        else{
           throw new RuntimeException("Illegal n_hidden_layer");
        }
        
        return result;
    }


    private int getNumOutputNeurons() {
        if(neurons==null)
        {
            return 0;
        }

        return neurons[neurons.length-1].length;
    }


}
