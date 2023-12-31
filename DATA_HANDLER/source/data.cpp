#include "data.hpp"

data::data()
{
    feature_vector = new std::vector<uint8_t>;
}

data::~data()
{
    delete feature_vector;
}
void data::set_feature_vector(std::vector<uint8_t> * vect)
{
    feature_vector = vect;
}

void data::set_feature_vector(std::vector<double> * vect)
{
    normalized_feature_vector = vect;
}

void data::append_to_feature_vector(uint8_t val)
{
    feature_vector->push_back(val);
}

void data::append_to_feature_vector(double val)
{
    normalized_feature_vector->push_back(val);
}

void data::set_feature_vector_val(int idx, double val)
{
    (*normalized_feature_vector)[idx] = val;
}

void data::set_class_vector(int count)
{
    class_vector = new std::vector<int>(count, 0);
    (*class_vector)[label] = 1;
}

void data::set_label(uint8_t val)
{
    label = val;
}
void data::set_enumeratedlabel(int val)
{
    enum_label = val;
}

void data::set_distance(double val)
{
    distance = val;
}

double data::get_distance()
{
    return distance;
}

int data::get_feature_vector_size()
{
    return feature_vector->size();
}
uint8_t data::get_label()
{
    return label;
}
uint8_t data::get_enumeratedlabel()
{
    return enum_label;
}

std::vector<uint8_t> * data::get_feature_vector()
{
    return feature_vector;
}

std::vector<double> * data::get_normalized_feature_vector()
{
    return normalized_feature_vector;
}
std::vector<int> * data::get_class_vector()
{
    return class_vector;
}