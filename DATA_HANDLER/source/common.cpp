#include "common.hpp"

void common::set_training_data(std::vector<data *> *vect)
{
    training_data = vect;
}
void common::set_test_data(std::vector<data *> *vect)
{
    test_data = vect;
}
void common::set_validation_data(std::vector<data *> *vect)
{
    validation_data = vect;
}