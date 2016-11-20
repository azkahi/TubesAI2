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
        breader = new BufferedReader(new FileReader("src\\main\\iris.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();
        FeedForwardNeuralNetworkAlgorithm FFNN = new FeedForwardNeuralNetworkAlgorithm(inputTrain);
        FFNN.buildModel(1, 5);
        double[] arr = inputTrain.get(0).toDoubleArray();
        FFNN.setInputLayer(arr);
        FFNN.printModel();
        
        System.out.println((FFNN.getNeurons())[1][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][2].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][3].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][4].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][2].getWeights().toString());
        
        double[] arro = {1,1,1,1,1};
        FFNN.determineOutput(inputTrain.get(0));
        System.out.println(FFNN.getClassOutputValues());
        FFNN.printModel();
        
        System.out.println((FFNN.getNeurons())[1][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][2].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][3].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][4].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][2].getWeights().toString());
        
        double err[] = FFNN.countOutputError(inputTrain.get(0));
        System.out.println(inputTrain.get(0).classValue());
        System.out.println("errour out :");
        System.out.println(err[0]);
        System.out.println(err[1]);
        System.out.println(err[2]);
        
        FFNN.updateModel();
        FFNN.printModel();
        System.out.println((FFNN.getNeurons())[1][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][2].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][3].getWeights().toString());
        System.out.println((FFNN.getNeurons())[1][4].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][0].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][1].getWeights().toString());
        System.out.println((FFNN.getNeurons())[2][2].getWeights().toString());
        
        /*
        err = FFNN.countHiddenError(inputTrain.get(0));
        System.out.println("errour hidden :");
        System.out.println(err[0]);
        System.out.println(err[1]);
        System.out.println(err[2]);*/
    }
    
}
