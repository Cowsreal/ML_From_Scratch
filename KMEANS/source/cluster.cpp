#include "cluster.hpp"

cluster::cluster(data* initial_point)
{
    m_centroid = new std::vector<double>;             //In this constructor, we initialize all member data to provided initial point
    m_cluster_points = new std::vector<data *>;
    for(auto value: *(initial_point->get_feature_vector()))
    {
        m_centroid->push_back(value);
    }
    m_cluster_points->push_back(initial_point);
    m_class_counts[initial_point->get_label()] = 1;
    m_most_frequent_class = initial_point->get_label();
}

cluster::~cluster()
{
    delete m_centroid;
    delete m_cluster_points;
}

void cluster::add_to_cluster(data* point)
{
    int previousSize = m_cluster_points->size();
    m_cluster_points->push_back(point);
    for(int i = 0; i < m_centroid->size() - 1; i++)
    {
        double value = (*m_centroid)[i];
        value *= previousSize;
        value += (*point->get_feature_vector())[i];
        value /= (double) m_cluster_points->size();
        (*m_centroid)[i] = value;
    }
    if(m_class_counts.find(point->get_label()) == m_class_counts.end())
    {
        m_class_counts[point->get_label()] = 1;
    }
    else
    {
        m_class_counts[point->get_label()]++;
    }
    set_most_frequent_class();
}

void cluster::set_most_frequent_class()
{
    int best_class;
    int frequency = 0;
    for(auto pair: m_class_counts)
    {
        if(pair.second > frequency)
        {
            best_class = pair.first;
            frequency = pair.second;
        }
        m_most_frequent_class = best_class;
    }
}

int cluster::get_most_frequent_class()
{
    return m_most_frequent_class;
}
std::vector<double> * cluster::get_centroid()
{
    return m_centroid;
}