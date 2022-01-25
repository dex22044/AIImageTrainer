#pragma once

#include <bits/stdc++.h>
#include <cstdlib>

class NNLayer
{
public:
	int size;
	double* neurons;
	double* biases;
	double** weights;

	NNLayer(int size, int nextSize);
};

