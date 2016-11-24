package FeedForwardNeuralNetwork;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.Randomize;

public class FeedForwardNeuralNetwork extends AbstractClassifier implements java.io.Serializable
{
    private FeedForwardNeuralNetworkAlgorithm FFNN;
    private boolean normalized = false;
    
    public FeedForwardNeuralNetworkAlgorithm getAlg(){
        return FFNN;
    }
    
    public void Normalize() throws Exception{
        
    }
    
    public void Randomize() throws Exception{
        Randomize randomize = new Randomize();
        randomize.setInputFormat(FFNN.getInstances());
        buildClassifier(Filter.useFilter(FFNN.getInstances(), randomize));
    }
    
    public void NTB() throws Exception{
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setInputFormat(FFNN.getInstances());
        buildClassifier(Filter.useFilter(FFNN.getInstances(), nominalToBinary));
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances origin = new Instances(instances);
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);
        
        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        FFNN = new FeedForwardNeuralNetworkAlgorithm(instances);
        trainModel(instances,1,50);
    }
    
    
    public void trainModel(Instances instances, int hidden_layer, int hidden_neurons){
        //Initialize
        FFNN.buildModel(hidden_layer, hidden_neurons);
        //Start Training
        double[] input = null;
        int j = 1;
        //j itu buat ngatur banyaknya iterasi training
        int error = 0;
        while (FFNN.getSumError() > error && j <= 5000){
            for (int i = 0; i<instances.size(); i++){
                FFNN.clearModel();
                input = instances.get(i).toDoubleArray();
                FFNN.setInputLayer(input);
                FFNN.determineOutput(instances.get(i));
                FFNN.updateModel(instances.get(i));
            }
            if (j % 100 == 0) System.out.println(j);
            j++;
        }
    }
    

    
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        FFNN.setInputLayer(instance.toDoubleArray());
        return FFNN.countOutput(instance);
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }
    
    
    
}