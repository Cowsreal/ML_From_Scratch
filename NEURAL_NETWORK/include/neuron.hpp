#ifndef __NEURON_H
#define __NEURON_H

#include <cmath>
#include <vector>

class neuron
{
public:
    neuron(int);
    void initializeWeights(int);
    std::vector<double>* getWeights();
    double getOutput();
    double getDelta();

    void setOutput(double);
    void setDelta(double);

private:
    double m_output;
    double m_delta;
    std::vector<double> m_weights;
};

#endif