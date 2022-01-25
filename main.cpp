#include <cstdio>
#include <fstream>
#include <fcntl.h>

#include "NeuralNetwork.h"
#include "NNSaver.h"

using namespace std;

void saveImage(char* filename, int w, int h, unsigned char* image) {
    ofstream wrtPixmap(filename, fstream::binary);
    wrtPixmap << "P6\n" << w << " " << h << "\n255\n";
    wrtPixmap.write((char*)image, w * h * 3);
    wrtPixmap.flush();
    wrtPixmap.close();
}

int main(int argc, char** argv) {
    if(argc != 2 && argc != 3) return -1;
    std::srand(time(NULL));
    unsigned char* imgPtr = NULL;
    int imgW = 0;
    int imgH = 0;

    {
        ifstream ppmFile(argv[1]);
        string fmt;
        ppmFile >> fmt;
        int maxVal = 0;
        ppmFile >> imgW >> imgH >> maxVal;
        int imSizePixels = imgW * imgH;
        imgPtr = new unsigned char[imSizePixels * 3];
        printf("W: %d\tH: %d\tMaxVal: %d\n", imgW, imgH, maxVal);
        ppmFile.read((char*)imgPtr, 1);

        if(fmt == "P6") {
            printf("P6 fmt\n");
            ppmFile.read((char*)imgPtr, imSizePixels * 3);
        } else {
            printf("P3 fmt\n");
            for(int i = 0; i < imSizePixels; i++) {
                int r, g, b;
                ppmFile >> r >> g >> b;
                imgPtr[i * 3 + 0] = r;
                imgPtr[i * 3 + 1] = g;
                imgPtr[i * 3 + 2] = b;
            }
        }
    }

    saveImage("./input.ppm", imgW, imgH, imgPtr);

    int layerSizes[] { 2, 100, 100, 100, 100, 100, 100, 100, 3 };
    NeuralNetwork* nn = NULL;
    int startGen = 0;
    if(argc == 2) nn = new NeuralNetwork(1e-5, sizeof(layerSizes) / 4, layerSizes);
    else {
        startGen = atoi(argv[2]);
        char fn[256];
        snprintf(fn, 255, "./generations/%d.nnmod", startGen);
        nn = NNSaver::Load(fn);
    }

    double* inputs = new double[2];
    double* outputsT = new double[3];

    int generations = 100000;
    int batches = 512 * 128;

    unsigned char* texData = new unsigned char[512 * 512 * 3];
    vector<double> errSums;

    std::mt19937 rng(time(0));

    for(int gen = startGen; gen < generations; gen++) {
        float errSum = 0.0f;
        for(int batch = 0; batch < batches; batch++) {
            int x = rng() % imgW;
            int y = rng() % imgH;
            int pxCoord = (y * imgW) + x;
            double r = imgPtr[pxCoord * 3 + 0] / 255.0;
            double g = imgPtr[pxCoord * 3 + 1] / 255.0;
            double b = imgPtr[pxCoord * 3 + 2] / 255.0;

            inputs[0] = x * 1.0 / imgW - 0.5;
            inputs[1] = y * 1.0 / imgH - 0.5;
            outputsT[0] = r;
            outputsT[1] = g;
            outputsT[2] = b;

            double* outputs = nn->FeedForward(inputs);

            for(int i = 0; i < 3; i++) {
                double err = outputsT[i] - outputs[i];
                errSum += err * err;
            }

            nn->Backpropogation(outputsT);
        }
        printf("Generation %d:\n", gen + 1);
        printf("ErrSum = %5.4f\n", errSum);
        if(errSum < 7000) nn->learningRate = 10e-7;
        if(errSum < 3000) nn->learningRate = 10e-10;
        errSums.push_back(errSum);

        if((gen + 1) % 20 == 0) {
            printf("Saving...\n");

            char fn[256];
    
            for(int y = 0; y < 512; y++) {
                for(int x = 0; x < 512; x++) {
                    inputs[0] = x / 512.0 - 0.5;
                    inputs[1] = y / 512.0 - 0.5;
                    double* outputs = nn->FeedForward(inputs);
                    int pxCoord = (y * 512) + x;
                    texData[pxCoord * 3 + 0] = (outputs[0] * 255);
                    texData[pxCoord * 3 + 1] = (outputs[1] * 255);
                    texData[pxCoord * 3 + 2] = (outputs[2] * 255);
                }
            }

            snprintf(fn, 255, "./generations/%d.ppm", gen + 1);
            saveImage(fn, 512, 512, texData);

            snprintf(fn, 255, "./generations/%d.nnmod", gen + 1);
            NNSaver::Save(nn, fn);

            ofstream errSumFile("./graph.csv");
            for(double val : errSums) {
                errSumFile << val << "\n";
            }
            errSumFile.close();

            printf("Saved.\n");
        }
    }
    
    for(int y = 0; y < 512; y++) {
        printf("Line: %d\n", y);
        for(int x = 0; x < 512; x++) {
            inputs[0] = x / 512.0 - 0.5;
            inputs[1] = y / 512.0 - 0.5;
            double* outputs = nn->FeedForward(inputs);
            int pxCoord = (y * 512) + x;
            texData[pxCoord * 3 + 0] = (outputs[0] * 255);
            texData[pxCoord * 3 + 1] = (outputs[1] * 255);
            texData[pxCoord * 3 + 2] = (outputs[2] * 255);
        }
    }

    saveImage("./output.ppm", 512, 512, texData);

    //SDL_Quit();
}