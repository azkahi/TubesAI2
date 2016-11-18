/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetworkAlgorithm;
import FeedForwardNeuralNetwork.Neuron;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;

/**
 *
 * @author user-ari
 */
public class coba {
    public static void main(String[] args) throws Exception {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("C:\\Users\\Frys\\Desktop\\Java\\AI\\Weka\\TubesAI2\\src\\main\\iris.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();
        FeedForwardNeuralNetworkAlgorithm FFNN = new FeedForwardNeuralNetworkAlgorithm(inputTrain);
        FFNN.buildModel(1, 5);
        double[] arr = inputTrain.get(0).toDoubleArray();
        FFNN.setInputLayer(arr);
        FFNN.printModel();
        double[] arro = {1,1,1,1,1};
        double res[] = FFNN.countOutput(inputTrain.get(0));
        System.out.println(res[0]);
        System.out.println(res[1]);
        System.out.println(res[2]);
        FFNN.printModel();
        double err[] = FFNN.countOutputError(inputTrain.get(0));
        System.out.println(err[0]);
        System.out.println(err[1]);
        System.out.println(err[2]);
        
        err = FFNN.countHiddenError(inputTrain.get(0));
        System.out.println(err[0]);
        System.out.println(err[1]);
        System.out.println(err[2]);
    }
    
}
