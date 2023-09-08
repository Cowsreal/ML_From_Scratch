#include <random>
#include "neuron.hpp"

double generate_random_number(double min, double max)
{
    double random = (double) rand() / (double) RAND_MAX;
    return min + random * (max - min);
}

neuron::neuron(int prevLayerSize)
{
    initializeWeights(prevLayerSize);
}
void neuron::initializeWeights(int prevLayerSize)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);    //(mean, std)
    for(int i = 0; i < prevLayerSize + 1; i++)
    {
        m_weights.push_back(generate_random_number(-1.0, 1.0));
    }
}

std::vector<double>* neuron::getWeights()
{
    return &m_weights;
}

double neuron::getOutput()
{
    return m_output;
}
double neuron::getDelta()
{
    return m_delta;
}

void neuron::setOutput(double val)
{
    m_output = val;
}

void neuron::setDelta(double val)
{
    m_delta = val;
}