#ifndef __KNN_H
#define __KNN_H

#include "common.hpp"

class knn: public common
{
public:
    knn(int val);
    knn();
    ~knn();

    
    void set_k(int val);
    
    void find_knearest(data *query_point);
    int predict();
    double calculate_distance(data * query_point, data * input);
    double validate_performance();
    double test_performance();

private:
    int k;
    std::vector<data *> * neighbors;
};

#endif