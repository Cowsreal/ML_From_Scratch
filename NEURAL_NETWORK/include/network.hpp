#ifndef __NETWORK_H
#define __NETWORK_H

#include "layer.hpp"
#include "neuron.hpp"
#include "data.hpp"
#include "common.hpp"

class network : public common
{
public:
    network(std::vector<int> spec, int inputSize, int numClasses, double learningRate, int activationFunction);
    ~network();
    std::vector<double> fProp(data* data);
    double activate(std::vector<double> weight, std::vector<double> prev);
    double transfer(double);
    double transferDerivative(double);
    void backProp(data* data);
    void updateWeights(data* data);
    int predict(data* data);
    void train(int);
    double test();
    void validate();
    void setActivationFunc(int);

private:
    std::vector<layer *> m_layers;
    double m_learningRate;
    double m_testPerformance;
    int m_activationFunctions;     //[sigmoid, relu, leaky relu]        
};

#endif