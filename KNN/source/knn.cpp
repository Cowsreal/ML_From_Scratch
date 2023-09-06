#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <iostream>
#include "data_handler.hpp"
#include <thread>

knn::knn(int val)
{
    k = val;
}
knn::knn()
{

}
knn::~knn()
{

}

//Notes: If k ~ N, then the time complexity is O(N^2)
//If k is very small, then approximately, the time complexity is O(~N)
//However, we can sort the data in ascending order, and the time complexity of STL sort is O(NlogN), we then perform a linear search, but N < NlogN, so the overall time complexity is O(NlogN)

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();        //STL for max value of type double
    double previous_min = min;
    int idx;
    for(int i = 0; i < k; i++)
    {
        if(i == 0)
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, (*training_data)[j]);     //Calculate distance
                (*training_data)[j]->set_distance(distance);                                //and store it
                if(distance < min)
                {
                    min = distance;
                    idx = j;
                }
            }
        }
        else
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = (*training_data)[j]->get_distance();                          //Access previously calculated distance
                if(distance > previous_min && distance < min)
                {
                    min = distance;
                    idx = j;
                }
            }
        }
        neighbors->push_back((*training_data)[idx]);
        previous_min = min;
        min = std::numeric_limits<double>::max();
    }
}

void knn::set_training_data(std::vector<data *> *vect)
{
    training_data = vect;
}
void knn::set_test_data(std::vector<data *> *vect)
{
    test_data = vect;
}
void knn::set_validation_data(std::vector<data *> *vect)
{
    validation_data = vect;
}
void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> class_freq;
    for(int i = 0; i < neighbors->size(); i++)      //Calculate the frequency of each neighbor class first
    {
        if(class_freq.find((*neighbors)[i]->get_label()) == class_freq.end())
        {
            class_freq[(*neighbors)[i]->get_label()] = 1;
        }
        else
        {
            class_freq[(*neighbors)[i]->get_label()]++;
        }
    }
    int pred = 0;
    int max = 0;
    for(auto key : class_freq)          //Return the class with the highest frequency
    {
        if(key.second > max)
        {
            max = key.second;
            pred = key.first;
        }
    }
    delete neighbors;
    return pred;
}

double knn::calculate_distance(data *query_point, data *input)
{
    double distance = 0.0;
    if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        std::cout << "Vector size mismatch" << std::endl;
        exit(1);
    }
    #ifdef EUCLID
        std::vector<uint8_t>* query_vec = query_point->get_feature_vector();
        std::vector<uint8_t>* input_vec = input->get_feature_vector();
        for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
        {
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
double knn::validate_performance()
{
    int count = 0;          //Number of correct predictions
    int data_idx = 0;
    for(data *query_point : *validation_data)           //For each entry in the validation set(query_point)
    {
        find_knearest(query_point);                     //Find the k nearest neighbors from the training set to the query point
        int prediction = predict();                     //Find of those k nearest neighbors, pick out the one appears the most frequently
        if(prediction == query_point->get_label())   //If the prediction is correct
        {
            count++;
        }
        data_idx++;
        std::cout << "Current performance = " << ((double)count * 100)/(data_idx) << "%" << std::endl;
    }
    std::cout << "Validation performance = " << ((double)count * 100)/(data_idx) << "%" << " k = " << k << std::endl;
    return ((double)count * 100)/(data_idx);
}
double knn::test_performance()
{
    int count = 0;          //Number of correct predictions
    int data_idx = 0;
    for(data *query_point : *test_data)
    {
        find_knearest(query_point);
        if(predict() == query_point->get_label())   //If the prediction is correct
        {
            count++;
        }
    }
    std::cout << "Test performance = " << ((double)count * 100)/(test_data->size()) << "%" << std::endl;
    return ((double)count * 100)/(test_data->size());
}

int main()
{
    data_handler* dh = new data_handler();
    dh->read_feature_vector("../data/train-images.idx3-ubyte");
    dh->read_feature_labels("../data/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    knn* knearest = new knn(1);
    knearest->set_training_data(dh->get_training_data());
    knearest->set_test_data(dh->get_test_data());
    knearest->set_validation_data(dh->get_validation_data());
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for(int i = 1; i < 5; i++)
    {
        if(i == 1)
        {
            continue;
            knearest->set_k(i);
            performance = knearest->validate_performance();
            best_performance = performance;
        }
        else
        {
            knearest->set_k(i);
            performance = knearest->validate_performance();
            if(performance > best_performance)
            {
                best_performance = performance;
                best_k = i;
            }
        }
    }
    knearest->set_k(best_k);
    knearest->test_performance();
}