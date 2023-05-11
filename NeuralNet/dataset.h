#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>
using namespace std;

double generateTarget(double feature1, double feature2, double feature3, double feature4) {
    return feature1 * feature2 + feature3- feature4 * feature4;
}
int create_dataset(int samples, string out_path) {
    // Set random seed
    srand(time(NULL));

    // Open CSV file for writing
    ofstream file(out_path);

    // Write header row
    //file << "Feature1,Feature2,Feature3,Feature4,Target" << endl;

    // Generate dataset
    for (int i = 0; i < samples; i++) {
        double feature1 = (double) rand() / RAND_MAX;
        double feature2 = (double) rand() / RAND_MAX;
        double feature3 = (double) rand() / RAND_MAX;
        double feature4 = (double) rand() / RAND_MAX;
        double target = generateTarget(feature1, feature2, feature3, feature4);

        // Write row to CSV file
        file << feature1 << "," << feature2 << "," << feature3 << "," << feature4 << "," << target << endl;
    }

    // Close CSV file
    file.close();

    return 0;
}


