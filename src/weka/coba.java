/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author user-ari
 */
public class coba {
    public static void main (String[] args) throws IOException{
        // Create empty instance with three attribute values
        //Read iris.arff
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src\\weka\\iris.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();

        System.out.println("The instance: " + inputTrain.get(0).toString()); 
        System.out.println("2The instance: " + inputTrain.get(0).classAttribute());
        System.out.println("3The instance: " + inputTrain.get(0).classIndex());
        System.out.println("4The instance: " + inputTrain.get(0).numClasses());
        System.out.println("5The instance: " + inputTrain.get(0).numAttributes());
        System.out.println("6The instance: " + inputTrain.get(0).numValues());
    }
}
