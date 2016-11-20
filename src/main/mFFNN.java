/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetwork;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author user-ari
 */
public class mFFNN {
    public static void main(String[] args) throws Exception {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src\\main\\iris.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();
        System.out.println("mFFNN!!!\n\n");
        FeedForwardNeuralNetwork FFNN = new FeedForwardNeuralNetwork();
        Evaluation eval = new Evaluation(inputTrain);
        //FFNN.trainModel(inputTrain, 1, 5);
    }
}
