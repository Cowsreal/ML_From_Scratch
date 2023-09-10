#include "main.hpp"
#include <iostream>
#include <iomanip>

#ifdef DATA_HANDLER
int main()
{
    data *bye = new data();
    cluster *hi = new cluster(bye);
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../data/train-images.idx3-ubyte");
    dh->read_feature_labels("../data/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    delete dh;
    return 0;
}
#endif

#ifdef KNN
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
    for(int i = 1; i < 5  ; i++)
    {
        if(i == 1)
        {
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
#endif

#ifdef KMEANS
int main()
{
    data_handler* dh = new data_handler();
    dh->read_feature_vector("../data/train-images.idx3-ubyte");
    dh->read_feature_labels("../data/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for(int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.1; k++)
    {
        kmeans* km = new kmeans(k);
        km->set_training_data(dh->get_training_data());
        km->set_validation_data(dh->get_validation_data());
        km->set_test_data(dh->get_test_data());
        km->init_clusters();
        km->train();
        performance = km->validate();
        std::cout << "Current Performance: " << std::setprecision(2) << std::fixed << performance << "%, k = " << k <<std::endl;
        if(performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }
    kmeans* km = new kmeans(best_k);
    km->set_training_data(dh->get_training_data());
    km->set_validation_data(dh->get_validation_data());
    km->set_test_data(dh->get_test_data());
    km->init_clusters();
    performance = km->test();
    std::cout << "Test Performance: " << std::setprecision(2) << std::fixed << performance << "%, k = " << best_k <<std::endl;
}
#endif

#ifdef NEURAL_NETWORK
int main()
{
    data_handler *dh = new data_handler();
#ifdef MNIST
    dh->read_feature_vector("../data/train-images.idx3-ubyte");
    dh->read_feature_labels("../data/train-labels.idx1-ubyte");
    dh->count_classes();

#elif IRIS
    dh->read_csv("../data/iris.data", ",");
#endif
    dh->split_data();
    std::vector<int> hiddenLayers = {10};
    int input = (*(dh->get_training_data()))[0]->get_normalized_feature_vector()->size();
    int output = dh->get_class_counts();
    std::cout << "Input size: " << input << std::endl;
    std::cout << "Output size: " << output << std::endl;
    auto lambda = [&]() {
        network* net = new network(
            hiddenLayers, 
            input, 
            output,
            0.001, 0);
        net->set_training_data(dh->get_training_data());
        net->set_test_data(dh->get_test_data());
        net->set_validation_data(dh->get_validation_data());
        net->train(30);
        net->validate();
        std::cout << "Test Performance: " << std::setprecision(2) << std::fixed << net->test() << "%" << std::endl;
    };
    lambda();
}
#endif
