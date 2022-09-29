#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>

#include <boost\program_options.hpp>

// #include "simple_net.h"
#include "lenet5.h"

int main(int argc, const char *argv[])
{
    std::string mnist_dataset_path;
    std::size_t epoch_num, batch_size;

    try
    {
        boost::program_options::options_description train_options_desc("Model training options");
        train_options_desc.add_options()("help,h", "help guide")("path,p", boost::program_options::value<std::string>(&mnist_dataset_path)->required(), "path to MNIST dataset")("epoch,e", boost::program_options::value<std::size_t>(&epoch_num)->default_value(5), "number of epoches")("batch_size,bs", boost::program_options::value<std::size_t>(&batch_size)->default_value(32), "batch size");
        boost::program_options::variables_map vm;

        if (argc < 2)
        {
            std::cerr << train_options_desc << std::endl;
            return -1;
        }
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, train_options_desc), vm);

        if (vm.count("help") > 0)
        {
            std::cout << train_options_desc << std::endl;
            return -1;
        }
        boost::program_options::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    if (std::filesystem::exists(std::filesystem::path(mnist_dataset_path).append("train-images-idx3-ubyte")) == false)
    {
        std::cout << "MNIST dataset path check failed! check path!" << std::endl;
        std::cout << "ERROR: " << mnist_dataset_path << "/train-images-idx3-ubyte does not exist!" << std::endl;
        return -1;
    }
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        std::cout << torch::cuda::device_count() << " cuda is available, use GPU." << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "cuda is NOT available, use CPU." << std::endl;
    }
    auto train_data_set = torch::data::datasets::MNIST(mnist_dataset_path, torch::data::datasets::MNIST::Mode::kTrain)
                              .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                              .map(torch::data::transforms::Stack<>());

    const std::size_t train_dataset_size = train_data_set.size().value();

    std::cout << "MINST dataset loaded, " << train_dataset_size << " training samples found." << std::endl;

    constexpr double learning_rate = 1e-2;

    auto train_loader = torch::data::make_data_loader(std::move(train_data_set), batch_size);

    // SimpleNet model(28 * 28, 300, 100, 10);
    LeNet5 model(28);
    model->to(device);
    auto criterion = torch::nn::CrossEntropyLoss();

    auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9));

    model->train();
    std::cout << "start training with setting: [epoch: " << epoch_num << ", batch size: " << batch_size << ", learing rate: " << learning_rate << "]" << std::endl;
    auto time_start = std::chrono::system_clock::now();
    for (int i = 1; i <= epoch_num; i++)
    {
        double sum_loss = 0.0;
        int train_correct = 0;
        for (auto &batch : *train_loader)
        {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);
            optimizer.zero_grad();
            auto outputs = model(inputs);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            sum_loss += loss.item().toDouble();
            auto [value, id] = torch::max(outputs.data(), 1);
            train_correct += torch::sum(id == labels).item().toInt();
        }
        std::cout << "[" << i << " / " << epoch_num << "]"
                  << " loss: " << sum_loss / (train_dataset_size / batch_size)
                  << ", correct: " << 100.0f * train_correct / train_dataset_size << std::endl;
    }
    auto time_end = std::chrono::system_clock::now();
    std::cout << "train time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << "ms" << std::endl; // 9s

    std::cout << "saving trained model..." << std::endl;
    torch::serialize::OutputArchive ar;
    model->save(ar);
    const std::string model_file = model->name() + ".pt";
    ar.save_to(model_file);
    std::cout << "trained model saved to " << model_file << std::endl;
    return 0;
}