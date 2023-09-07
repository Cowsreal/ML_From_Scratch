#include <cmath>
#include <limits>
#include <map>
#include <iostream>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <chrono>
#include "knn.hpp"
#include "data_handler.hpp"

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

bool comp(data* a, data* b)                                             //Comparator function for sorting by distances
{
    return a->get_distance() < b->get_distance();
}

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();                    //STL for max value of type double
    double previous_min = min;
    int idx;
for(int i = 0; i < k; i++)                                              //Calculate distance between query_point and every training data point
    {
        for(int j = 0; j < training_data->size(); j++)
        {
            double distance = calculate_distance(query_point, (*training_data)[j]);
            (*training_data)[j]->set_distance(distance);
        }
    }
    std::sort((*training_data).begin(), (*training_data).end(), comp);  //Sort the training data by distance
    for(int i = 0; i < k; i++)                                          //Loop thru k nearest neighbors
    {
        neighbors->push_back((*training_data)[i]);                      //Add the k nearest neighbors to neighbors vector
    }
    /*
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
    }*/
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
    std::vector<uint8_t>* query_vec = query_point->get_feature_vector();
    std::vector<uint8_t>* input_vec = input->get_feature_vector();
    #ifdef EUCLID       //L2 NORM
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
    #elif defined MANHATTAN     //L1 NORM.
        for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
        {
            distance += abs((*query_vec)[i] - (*input_vec)[i]);
        }
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
        std::cout << "Current performance = " << std::setprecision(2) << std::fixed << ((double)count * 100)/(data_idx) << "%" << "(" << count << "/" << data_idx << "), k = " << k;
        auto now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        // Convert the time to a string and print it
        std::cout << "\nCurrent time is " << std::ctime(&current_time) << "\r" << "\r" << std::flush;
    }
    std::cout << "Validation performance = " << std::setprecision(2) << std::fixed << ((double)count * 100)/(data_idx) << "%" << " k = " << k << std::endl;
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
    std::cout << "Test performance = " << std::setprecision(2) << std::fixed << ((double)count * 100)/(test_data->size()) << "%" << std::endl;
    return ((double)count * 100)/(test_data->size());
}