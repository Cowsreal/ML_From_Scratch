#ifndef __COMMON_H
#define __COMMON_H

#include "data.hpp"
#include <vector>

class common
{
protected:
    std::vector<data *> * training_data;
    std::vector<data *> * test_data;
    std::vector<data *> * validation_data;
public:
    void set_training_data(std::vector<data *> * vect);
    void set_test_data(std::vector<data *> * vect);
    void set_validation_data(std::vector<data *> * vect);

};

#endif