#ifndef __LAYER_H
#define __LAYER_H

#include "neuron.hpp"
#include <stdint.h>
#include <vector>

class layer
{
public:
    layer(int, int);
    ~layer();

    std::vector<double> getLayerOutputs();
    int getLayerSize();
    std::vector<neuron *> getNeurons();

private:
    int m_layerSize;
    std::vector<neuron *> m_neurons;
    std::vector<double> m_layerOutput;

};

#endif