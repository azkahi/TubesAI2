/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetwork;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author user-ari
 */
public class mFFNN {
    
   void saveModel(Classifier C, String namaFile) throws Exception {
        //SAVE 
         // serialize model
                ObjectOutputStream oos = new ObjectOutputStream(
                                           new FileOutputStream(namaFile));
                oos.writeObject(C);
                oos.flush();
                oos.close();
    }
    
    public static void main(String[] args) throws Exception {
        mFFNN m = new mFFNN();
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src\\main\\iris.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();
        System.out.println("mFFNN!!!\n\n");
        FeedForwardNeuralNetwork FFNN = new FeedForwardNeuralNetwork();
        Evaluation eval = new Evaluation(inputTrain);
        //FFNN.trainModel(inputTrain, 1, 5);
        FFNN.buildClassifier(inputTrain);
        eval.evaluateModel(FFNN,inputTrain);
        
        //OUTPUT
        Scanner scan = new Scanner(System.in);
        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
        System.out.println(eval.toMatrixString("===Confusion matrix==="));
        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
        System.out.println("\nDo you want to save this model(1/0)? ");
        int c = scan.nextInt();
        if (c == 1 ){
             System.out.print("Please enter your file name (*.model) : ");
             String infile = scan.next();
             m.saveModel(FFNN,infile);
        }
        else {
            System.out.print("Model not saved.");
        }
    }
}
