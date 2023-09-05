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
    void set_label(uint8_t);
    void set_enumeratedlabel(int);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumeratedlabel();

    std::vector<uint8_t> * get_feature_vector();

private:
    std::vector<uint8_t> * feature_vector;
    uint8_t label;
    int enum_label;
};


#endif