#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <torch/torch.h>
#include "csv_to_tensor.h"
#include "dataset.h"
using namespace std;
using namespace torch;

float predict(Tensor features, torch::nn::Sequential net){
    Tensor predictions = net->forward(features);
    float prediction = predictions[0].item<float>();
    return prediction;

}
torch::nn::Sequential neural_net(Tensor features, Tensor target, int epochs ){
    // Define the neural network architecture
    nn::Sequential net(
        nn::Linear(4, 16),
        nn::ReLU(),
        nn::Linear(16, 32),
        nn::ReLU(),
        nn::Linear(32, 16),
        nn::ReLU(),
        nn::Linear(16, 1)
    );
// Change the data type of the weight tensors to kDouble

    // Define the loss function and optimizer
    nn::MSELoss loss_fn;
    optim::SGD optimizer(net->parameters(), 0.01);

    //adjust dtype of input and target to expected input type of network
    features = features.to(net->parameters()[0].dtype());
    target = target.to(net->parameters()[0].dtype());

    // Train the neural network
        cout << "Neural Net: " << endl;
        cout << net << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        Tensor output = net->forward(features);
        Tensor loss = loss_fn(output, target);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Print the loss every 100 epochs
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << endl;
        }
    }

    // Save the trained model to file
    torch::save(net, "models/model.pt");

    cout << "Training complete!" << endl;

    return net;
}

int main() {
   string path="data/dataset.csv";
   int epochs = 20000;
   create_dataset(1000,path);
   Tensor tensor=c_to_tensor(path);
   
   Tensor features = tensor.narrow(1, 0, 4);
   Tensor target = tensor.narrow(1, 4, 1);

    torch::nn::Sequential net= neural_net(features, target, epochs);

    Tensor test_sample= torch::tensor({{0.9,0.8, 0.2, 0.1}});
    cout << predict(test_sample, net) << endl;

}


