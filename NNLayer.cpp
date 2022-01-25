#include "NNLayer.h"

NNLayer::NNLayer(int size, int nextSize) : size(size) {
	neurons = (double*)calloc(size, 8);
	biases = (double*)calloc(size, 8);
	weights = (double**)calloc(size, sizeof(double*));
	
	for (int i = 0; i < size; i++) {
		weights[i] = (double*)calloc(nextSize, 8);
	}
}