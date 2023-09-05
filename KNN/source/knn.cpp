#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <iostream>
#include "data_handler.hpp"

knn::knn(int k)
{
    k = k;
}
knn::knn()
{

}
knn::~knn()
{

}

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int idx = 0;
    for(int i = 0; i < k; i++)
    {
        if(i==0)
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, (*training_data)[j]);
                
            }
        }
    }
}
void knn::set_training_data(std::vector<data *> * vect)
{
    training_data = vect;
}
void knn::set_test_data(std::vector<data *> * vect)
{
    test_data = vect;
}
void knn::set_validation_data(std::vector<data *> * vect)
{
    validation_data = vect;
}
void knn::set_k(int k)
{
    k = k;
}

int knn::predict();
double knn::calculate_distance(data * query_point, data * input)
{
    #define EUCLID
    double distance = 0.0;
    if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        std::cout << "Vector size mismatch" << std::endl;
        exit(1);
    }
    #ifdef EUCLID
        for(unsigned i; i < query_point->get_feature_vector_size(); i++)
        {
            std::vector<uint8_t> * query_vec = query_point->get_feature_vector();
            std::vector<uint8_t> * input_vec = input->get_feature_vector();
            if(query_vec && input_vec)
            {
                distance += pow((*query_vec)[i] - (*input_vec)[i], 2);
            }
            else
            {
                if(query_vec == nullptr)
                {
                    std::cout << "query_vec is nullptr" << std::endl;
                }
                else
                {
                    std::cout << "input_vec is nullptr" << std::endl;
                }
                exit(1);
            }
        }
        distance = sqrt(distance);
    #elif define MANHATTAN
    #endif
    return distance;
}
double knn::validate_performance();
double knn::test_performance();