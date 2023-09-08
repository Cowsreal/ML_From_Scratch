#include <numeric>
#include <algorithm>
#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"

network::network(std::vector<int> spec, int inputSize, int numClasses, double learningRate, int activationFunction = 0)
{
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)                                                  // input layer
        {
            m_layers.push_back(new layer(inputSize, spec[i]));
        }
        else
        {                                                           // hidden layers
            m_layers.push_back(new layer(spec[i-1], spec[i]));
        }
    }
    m_layers.push_back(new layer(spec[spec.size()-1], numClasses)); // output layer
    m_learningRate = learningRate;
    m_activationFunctions = activationFunction;
}

network::~network()
{
    for(int i = 0; i < m_layers.size(); i++)
    {
        delete m_layers[i];
    }
}

double network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights[weights.size()-1];      //bias
    for(int i = 0; i < weights.size() - 1; i++)
    {
        activation += input[i] * weights[i];
    }
    return activation;
}

double network::transfer(double activation)
{
    switch(m_activationFunctions)
    {
        case 0:                                                         //sigmoid
            return 1.0 / (1.0 + exp(-activation));                      
        case 1:                                                             //relu
            if(activation > 0)
            {
                return activation;
            }
            else
            {
                return 0;
            }
        case 2:                                                             //leaky relu
            if(activation > 0)
            {
                return activation;
            }
            else
            {
                return 0.01 * activation;
            }
        default:
            {
                exit(1);
            }
    }
}   

double network::transferDerivative(double output)
{
    switch(m_activationFunctions)
    {
        case 0:                                                         //sigmoid
            return output * (1 - output);
        case 1:                                                             //relu
            if(output > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        case 2:                                                             //leaky relu
            if(output > 0)
            {
                return 1;
            }
            else
            {
                return 0.01;
            }
        default:
        {
            exit(1);
        }
    }
}

std::vector<double> network::fProp(data* data)
{
    std::vector<double> inputs = *(data->get_normalized_feature_vector());
    for(int i = 0; i < m_layers.size(); i++)
    {
        layer* layer = m_layers[i];
        std::vector<double> newInputs;
        for(neuron * n : layer->getNeurons())
        {
            double activation = activate(*(n->getWeights()), inputs);      //Calculate dot products
            n->setOutput(transfer(activation));                         //Set activation
            newInputs.push_back(n->getOutput());                        //Add to new input vector
        }
        inputs = newInputs;
    }
    return inputs;      //Input after going thru each layer
}

void network::backProp(data* data)
{
    for(int i = m_layers.size()-1; i >= 0; i--)
    {
        layer* layer = m_layers[i];
        std::vector<double> errors;
        if(i != m_layers.size()-1)
        {
            for(int j = 0; j < layer->getNeurons().size(); j++)
            {
                double error = 0.0;
                for(neuron * n : m_layers[i+1]->getNeurons())
                {
                    error += ((*(n->getWeights()))[j] * n->getDelta());
                }
                errors.push_back(error);
            }
        }
        else
        {
            for(int j = 0; j < layer->getNeurons().size(); j++)
            {
                neuron* n  = layer->getNeurons()[j];
                errors.push_back((double)(*(data->get_class_vector()))[j] - n->getOutput());        //Loss
            }
        }
        for(int j = 0; j < layer->getNeurons().size(); j++)
        {
            neuron* n  = layer->getNeurons()[j];
            n->setDelta(errors[j] * transferDerivative(n->getOutput()));        //Keep track of gradients
        }
    }
}

void network::updateWeights(data* data)
{
    std::vector<double> inputs = *(data->get_normalized_feature_vector());
    for(int i = 0; i < m_layers.size(); i++)
    {
        if(i != 0)
        {
            for(neuron* n : m_layers[i-1]->getNeurons())
            {
                inputs.push_back(n->getOutput());
            }
        }
        for(neuron* n : m_layers[i]->getNeurons())
        {
            for(int j = 0; j < inputs.size(); j++)
            {
                (*(n->getWeights()))[j] += m_learningRate * inputs[j] * n->getDelta();
            }
            (*(n->getWeights()))[n->getWeights()->size()-1] += m_learningRate * n->getDelta();
        }
        inputs.clear();
    }
}

int network::predict(data* data)
{
    std::vector<double> outputs = fProp(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void network::train(int numEpochs)
{
    for(int i = 0; i < numEpochs; i++)
    {
        double sumError = 0.0;
        for(data* d : *training_data)
        {
            std::vector<double> outputs = fProp(d);
            std::vector<int> expected = *(d->get_class_vector());
            double tempErrorSum = 0.0;
            for(int j = 0; j < outputs.size(); j++)
            {
                tempErrorSum += std::pow((double)outputs[j] - expected[j], 2);
            }
            sumError += tempErrorSum;
            backProp(d);
            updateWeights(d);
        }
        std::cout << "Epoch: " << i << " Error: " << sumError << std::endl;
    }
}

double network::test()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(data* d : *test_data)
    {
        count++;
        int index = predict(d);
        if((*d->get_class_vector())[index] == 1.0)
        {
            numCorrect++;
        }
    }
    return numCorrect / count;
}

void network::validate()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(data* d : *validation_data)
    {
        count++;
        int index = predict(d);
        if((*d->get_class_vector())[index] == 1.0)
        {
            numCorrect++;
        }
    }
    std::cout << "Validation Accuracy: " << numCorrect / count << std::endl;
}


void network::setActivationFunc(int idx)
{
    m_activationFunctions = idx;
}