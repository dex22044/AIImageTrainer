#pragma once

#include <bits/stdc++.h>
#include <string>

#include "NNLayer.h"
#include "NNTools.h"

class NeuralNetwork
{
public:
	double learningRate;
	int layers;
	NNLayer** nnlayers;

	NeuralNetwork(double learningRate, int layers, int* sizes);

	double* FeedForward(double* inputData);
	void FeedForward(double* inputData, double* outputs);
	void Backpropogation(double* targets);
};

