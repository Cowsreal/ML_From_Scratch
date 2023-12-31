#include "data_handler.hpp"
#include <algorithm>
#include <random>

data_handler::data_handler() : TRAIN_SET_PERCENT(0.75), TEST_SET_PERCENT(0.2), VALIDATION_SET_PERCENT(0.05)
{
    data_array = new std::vector<data *>;
    test_data = new std::vector<data *>;
    training_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}
data_handler::~data_handler()
{
    for(int i = 0; i < data_array->size(); i++)
    {
        delete (*data_array)[i];
        (*data_array)[i] = nullptr;
    }
    delete [] data_array;
    delete [] test_data;
    delete [] training_data;
    delete [] validation_data;
}

void data_handler::read_csv(std::string path, std::string delimiter)
{
    num_classes = 0;
    std::ifstream file(path.c_str());
    std::string line;
    while(std::getline(file, line))     //Parse each line
    {
        if(line.length() == 0)          //Empty line
        {
            continue;
        }
        data *d = new data();
        d->set_feature_vector(new std::vector<double>());
        size_t position = 0;
        std::string token;
        while((position = line.find_first_of(delimiter))!= std::string::npos)       //While there exists a delimiter in the current line
        {
            token = line.substr(0, position);                                       //Insert the current token into the current data
            d->append_to_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        if(class_map_str.find(line) != class_map_str.end()) //Now, all that's left in line is the class label
        {
            d->set_label(class_map_str[line]);
        }
        else
        {
            class_map_str[line] = num_classes;
            d->set_label(class_map_str[line]);
            num_classes++;
        }
        data_array->push_back(d);
    }
    for(data *d : *data_array)
    {
        d->set_class_vector(num_classes);
    }
    feature_vector_size = (*data_array)[0]->get_normalized_feature_vector()->size();
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4];// MAGIC NUMBER|NUM IMAGES|ROW SIZE|COLUMN SIZE, header information
    unsigned char bytes[4]; // temp store bytes read from header
    FILE *f = fopen(path.c_str(), "rb");    //pointer to FILE for file io
    if(f)
    {
        for(int i = 0; i < 4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);    //Convert header information to little endian and store in header array
            }
        }
        std::cout << "Done getting input file header.\n";
        uint32_t image_size = header[2] * header[3];
        for(int i = 0; i < header[1]; i++)  //Read the rest of the file (images)
        {
            data *d = new data();
            d->set_feature_vector(new std::vector<uint8_t>());
            uint8_t element[1];
            for(int j = 0; j < image_size; j++)
            {
                if(fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    std::cout << "Error reading from file.\n";
                    exit(1);
                }
                
            }
            data_array->push_back(d);
        }
        std::cout << "Done reading input file.\n";
        feature_vector_size = (*data_array)[0]->get_feature_vector()->size();
        std::cout << "Successfully read and stored feature vectors.\n";
    }
    else
    {
        std::cout << "Could not find file.\n";
        exit(1);
    }
}
void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2];// MAGIC NUMBER|NUM IMAGES|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if(f)
    {
        for(int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Done getting label file header.\n";
        for(int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, f))
            {
                (*data_array)[i]->set_label(element[0]);
            }
            else
            {
                std::cout << "Error reading from file.\n";
                exit(1);
            }
        }
        std::cout << "Successfully read and stored label.\n";
    }
    else
    {
        std::cout << "Could not find file.\n";
        exit(1);
    }
}
void data_handler::split_data()
{
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int validate_size = data_array->size() * VALIDATION_SET_PERCENT;

    std::random_device rd;              //Random number generator
    std::mt19937 g(rd());
    std::shuffle(data_array->begin(), data_array->end(), g);        //Shuffle the data array and then push pointers one by one to data sets

    int count = 0;
    while(count < train_size)
    {
        training_data->push_back((*data_array)[count]);
        count++;
    }
    
    count = 0;
    while(count < test_size)
    {
        test_data->push_back((*data_array)[count]);
        count++;
    }
    
    count = 0;
    while(count < validate_size)
    {
        validation_data->push_back((*data_array)[count]);
        count++;
    }
    std::cout << "Training Data Size: " << training_data->size() << "\n";
    std::cout << "Test Data Size: " << test_data->size() << "\n";
    std::cout << "Validation Data Size: " << validation_data->size() << "\n";
}
void data_handler::count_classes()
{
    int count = 0;
    for(unsigned i = 0; i < data_array->size(); i++)        //Loops over entire data set
    {
        if(class_map_int.find((*data_array)[i]->get_label()) == class_map_int.end())        //If label is not already in map
        {
            class_map_int[(*data_array)[i]->get_label()] = count;       //Add label to map
            (*data_array)[i]->set_enumeratedlabel(count);       //Set label to enumerated version of label
            count++;
        }
        else
        {
        (*data_array)[i]->set_enumeratedlabel(class_map_int[(*data_array)[i]->get_label()]);       //Set
        }  
    }
    num_classes = count;
    for(data * data : *data_array)
    {
        data->set_class_vector(num_classes);
    }
    std::cout << "Successfully extracted " << num_classes << " unique classes.\n";
    normalize();
}

void data_handler::set_train_percent(double train_percent)
{
    TRAIN_SET_PERCENT = train_percent;
}
void data_handler::set_test_percent(double test_percent)
{
    TEST_SET_PERCENT = test_percent;
}
void data_handler::set_validation_percent(double validation_percent)
{
    VALIDATION_SET_PERCENT = validation_percent;
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes)
{
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

std::vector<data *> * data_handler::get_training_data()
{
    return training_data;
}
std::vector<data *> * data_handler::get_test_data()
{
    return test_data;
}
std::vector<data *> * data_handler::get_validation_data()
{
    return validation_data;
}

int data_handler::get_class_counts()
{
    return num_classes;
}

void data_handler::normalize()
{
    std::vector<double> mins, maxs;
    data *d = (*data_array)[0];
    for(auto val : *d->get_feature_vector())
    {
        mins.push_back(val);                            //Arbitrarily initialize mins and maxs vector with data_array[0]
        maxs.push_back(val);
    }
    
    for(int i = 1; i < data_array->size(); i++)
    {
        d = (*data_array)[i];                           //d is the ith data object in data_array
        for(int j = 0; j < d->get_feature_vector_size(); j++)       //Iterate thru each feature inside the vector
        {
            double value = (double) (*d->get_feature_vector())[j];  //Get value of the jth feature
            if(value < mins[j])                                     //Update mins and maxs vectors
            {
                mins[j] = value; 
            }
            if(value > maxs[j])                         
            {
                maxs[j] = value;
            }
        }
    }
    for(int i = 0; i < data_array->size(); i++)
    {
        int size = (*data_array)[i]->get_feature_vector_size();
        std::vector<double>* vect1 = new std::vector<double>(size, 0);
        (*data_array)[i]->set_feature_vector(vect1);
        (*data_array)[i]->set_class_vector(num_classes);
        auto vect = (*data_array)[i]->get_feature_vector();
        for(int j = 0; j < size; j++)
        {
            if(maxs[j] - mins[j] == 0)
            {
                (*data_array)[i]->set_feature_vector_val(j, 0.0);
            }
            else
            {
                (*data_array)[i]->set_feature_vector_val(j, (double)(vect->at(j) - mins[j])/(maxs[j]-mins[j]));
            }
        }
    }
}