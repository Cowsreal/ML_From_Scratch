#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <cstdint>
#include <iostream>

class data
{
public:
    data();
    ~data();
    
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_feature_vector(std::vector<double> * vect);
    void append_to_feature_vector(double val);
    void set_class_vector(int count);
    void set_label(uint8_t);
    void set_enumeratedlabel(int);
    void set_distance(double val);

    double get_distance();
    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumeratedlabel();
    std::vector<uint8_t> * get_feature_vector();
    std::vector<double> * get_normalized_feature_vector();
    std::vector<int> * get_class_vector();

private:
    std::vector<uint8_t> * feature_vector;
    std::vector<double> * normalized_feature_vector;
    std::vector<int> * class_vector;
    uint8_t label;
    int enum_label;
    double distance;
};


#endif