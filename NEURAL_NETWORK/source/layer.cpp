#include "layer.hpp"

layer::layer(int prevLayerSize, int currLayerSize)
{
    for(int i = 0; i < currLayerSize; i++)
    {
        m_neurons.push_back(new neuron(prevLayerSize));
    }
    m_layerSize = currLayerSize;
}

layer::~layer()
{
    for(int i = 0; i < m_layerSize; i++)
    {
        delete m_neurons[i];
    }
}

std::vector<double> layer::getLayerOutputs()
{
    return m_layerOutput;
}
int layer::getLayerSize()
{
    return m_layerSize;
}
std::vector<neuron *> layer::getNeurons()
{
    return m_neurons;
}