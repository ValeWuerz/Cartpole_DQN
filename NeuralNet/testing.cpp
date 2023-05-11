#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <torch/torch.h>
using namespace std;
using namespace torch;

int main(){
std::vector<std::vector<double>> data = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0}
};
int rows = data.size();
int cols = data[0].size();
torch::Tensor tensor = torch::zeros({rows, cols}, torch::kDouble);

for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        tensor[i][j] = data[i][j];
    }
}

cout << tensor << endl;

}