#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include "data.hpp"

class data_handler
{
public:
    data_handler();
    ~data_handler();

    void read_csv(std::string path, std::string delimiter);
    void read_feature_vector(std::string path);
    void read_feature_labels(std::string path);
    void split_data();
    void count_classes();
    void set_train_percent(double train_percent);
    void set_test_percent(double test_percent);
    void set_validation_percent(double validation_percent);

    uint32_t convert_to_little_endian(const unsigned char* bytes);

    std::vector<data *> * get_training_data();
    std::vector<data *> * get_test_data();
    std::vector<data *> * get_validation_data();
    int get_class_counts();
    void normalize();

private:
    double TRAIN_SET_PERCENT;
    double TEST_SET_PERCENT;
    double VALIDATION_SET_PERCENT;

    std::vector<data *> * data_array;
    std::vector<data *> * training_data;
    std::vector<data *> * test_data;
    std::vector<data *> * validation_data;

    int num_classes;
    int feature_vector_size;
    std::map<uint8_t, int> class_map_int;
    std::map<std::string, int> class_map_str;
};


#endif