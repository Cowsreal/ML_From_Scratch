#ifndef __CLUSTER_H
#define __CLUSTER_H

#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_set>
#include "data_handler.hpp"

class cluster
{
public:
    cluster(data * initial_point);
    ~cluster();

    void add_to_cluster(data * point);
    void set_most_frequent_class();

    int get_most_frequent_class();
    std::vector<double> * get_centroid();

private:
    std::vector<double> * m_centroid;
    std::vector<data *> * m_cluster_points;
    std::map<int, int> m_class_counts;
    int m_most_frequent_class;
};

typedef cluster cluster_t;

#endif