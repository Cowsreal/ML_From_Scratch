#include "kmeans.hpp"
#include <algorithm>

kmeans::kmeans(int k)
{
    m_num_clusters = k;
    m_clusters = new std::vector<cluster_t *>;
    m_used_indices = new std::unordered_set<int>;
}

kmeans::~kmeans()
{
    for(auto cluster : *m_clusters)
    {
        delete cluster;
    }
    delete m_clusters;
    delete m_used_indices;
}

void kmeans::init_clusters()
{
    for(int i = 0; i < m_num_clusters; i++)
    {
        int index = rand() % training_data->size();
        while(m_used_indices->find(index) != m_used_indices->end())      //Keep randomizing until we find an unused index
        {
            index = rand() % training_data->size();
        }
        m_clusters->push_back(new cluster_t((*training_data)[index]));
        m_used_indices->insert(index);
    }
}

void kmeans::init_clusters_for_each_class()
{
    std::unordered_set<int> classes_used;
    for(int i = 0; i < training_data->size(); i++)
    {
        if(classes_used.find((*training_data)[i]->get_label()) == classes_used.end())
        {
            cluster_t *new_cluster = new cluster_t((*training_data)[i]);
            m_clusters->push_back(new_cluster);
            classes_used.insert((*training_data)[i]->get_label());
            m_used_indices->insert(i);                    //Still need to keep track of which indexes have been used
        }
    }
}

// O(N * k * 28*28), with 1<=k<=N, O(N^2)

void kmeans::train()
{
    int index = 0;
    while(m_used_indices->size() < training_data->size())
    {
        while(m_used_indices->find(index)!= m_used_indices->end())
        {
            index++;
        }
        double min_dist = std::numeric_limits<double>::max();           //Decide what cluster to add current point to
        int best_cluster = 0;
        for(int j = 0; j < m_clusters->size(); j++)
        {
            double current_dist = 0.0;
            #ifdef EUCLID
            current_dist = euclidean_distance((*m_clusters)[j]->get_centroid(), (*training_data)[index]);
            #endif
            #ifdef MANHATTAN
            current_dist = manhattan_distance((*m_clusters)[j]->get_centroid(), (*training_data)[index]);
            #endif
            if(current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        (*m_clusters)[best_cluster]->add_to_cluster((*training_data)[index]);
        m_used_indices->insert(index);
    }
}

double kmeans::euclidean_distance(std::vector<double>* centroid, data* point)       //l2 norm
{
    double distance = 0.0;
    for(int i = 0; i < centroid->size(); i++)
    {
        distance += pow((*centroid)[i] - (*point->get_feature_vector())[i], 2);
    }
    return sqrt(distance);
}

double kmeans::manhattan_distance(std::vector<double>* centroid, data* point)       //l1 norm
{
    double distance = 0.0;
    for(int i = 0; i < centroid->size(); i++)
    {
        distance += abs((*centroid)[i] - (*point->get_feature_vector())[i]);
    }
    return distance;
}

double kmeans::validate()
{
    double num_correct = 0;
    for(auto query_point : *validation_data)
    {
        double min_dist = std::numeric_limits<double>::max();           //Decide what cluster to add current point to
        int best_cluster = 0;
        for(int j = 0; j < m_clusters->size(); j++)
        {
            double current_dist = euclidean_distance((*m_clusters)[j]->get_centroid(), query_point);
            if(current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        if((*m_clusters)[best_cluster]->get_most_frequent_class() == query_point->get_label())
        {
            num_correct++;
        }
    }
    return (num_correct / (double) validation_data->size()) * 100.0;
}

double kmeans::test()
{
    double num_correct = 0;
    for(auto query_point : *test_data)
    {
        double min_dist = std::numeric_limits<double>::max();           //Decide what cluster to add current point to
        int best_cluster = 0;
        for(int j = 0; j < m_clusters->size(); j++)
        {
            double current_dist = euclidean_distance((*m_clusters)[j]->get_centroid(), query_point);
            if(current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        if((*m_clusters)[best_cluster]->get_most_frequent_class() == query_point->get_label())
        {
            num_correct++;
        }
    }
    return (num_correct / (double) test_data->size()) * 100.0;
}