#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <torch/torch.h>
using namespace std;
using namespace torch;


Tensor c_to_tensor(string path) {
    vector<vector<double>> data;
       ifstream file(path);
      string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;
        while (getline(ss, value, ',')) {
            double d = stod(value);
            row.push_back(d);
        }
        data.push_back(row);
    }
    file.close();
    int rows = data.size();
    int cols = data[0].size();
    torch::Tensor tensor = torch::zeros({rows, cols}, torch::kDouble);

for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        tensor[i][j] = data[i][j];
    }
}
   // cout << tensor << endl;
    return tensor;   
}


