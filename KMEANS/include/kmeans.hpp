#ifndef __KMEANS_H
#define __KMEANS_H

#include "common.hpp"
#include "data_handler.hpp"
#include "cluster.hpp"
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>

class kmeans: public common
{
public:
    kmeans(int k);
    ~kmeans();

    void init_clusters();
    void init_clusters_for_each_class();
    void train();
    double euclidean_distance(std::vector<double>* centroid, data* point);
    double manhattan_distance(std::vector<double>* centroid, data* point);
    double validate();
    double test();
private:
    int m_num_clusters;
    std::vector<cluster_t *>* m_clusters;
    std::unordered_set<int>* m_used_indices;
};

#endif