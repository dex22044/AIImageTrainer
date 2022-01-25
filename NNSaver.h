#pragma once

#include <bits/stdc++.h>
#include <fstream>

#include "NeuralNetwork.h"

using namespace std;

class NNSaver
{
public:
	static void Save(NeuralNetwork* nn, string path);
	static NeuralNetwork* Load(string path);
};
