#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(double learningRate, int layers, int* sizes) : layers(layers), learningRate(learningRate) {
	nnlayers = new NNLayer*[layers];

	for (int i = 0; i < layers; i++) {
		int nextSize = (i < layers - 1) ? sizes[i + 1] : 0;
		nnlayers[i] = new NNLayer(sizes[i], nextSize);
		for (int j = 0; j < sizes[i]; j++) {
			nnlayers[i]->biases[j] = (std::rand() * 1.0 / RAND_MAX) * 2.0 - 1.0;
			for (int k = 0; k < nextSize; k++) {
				nnlayers[i]->weights[j][k] = (std::rand() * 1.0 / RAND_MAX) * 2.0 - 1.0;
			}
		}
	}
}

double* NeuralNetwork::FeedForward(double* inputData) {
	memcpy(nnlayers[0]->neurons, inputData, nnlayers[0]->size * 8);

	for (int i = 1; i < layers; i++) {
		NNLayer* l = nnlayers[i - 1];
		NNLayer* l1 = nnlayers[i];
		for (int j = 0; j < l1->size; j++) {
			l1->neurons[j] = 0;
			for (int k = 0; k < l->size; k++) {
				l1->neurons[j] += l->neurons[k] * l->weights[k][j];
			}
			l1->neurons[j] += l1->biases[j];
			l1->neurons[j] = sigmoid(l1->neurons[j]);
		}
	}

	return nnlayers[layers - 1]->neurons;
}

void NeuralNetwork::FeedForward(double* inputData, double* outputs) {
	memcpy(nnlayers[0]->neurons, inputData, nnlayers[0]->size * 8);

	for (int i = 1; i < layers; i++) {
		NNLayer* l = nnlayers[i - 1];
		NNLayer* l1 = nnlayers[i];
		for (int j = 0; j < l1->size; j++) {
			l1->neurons[j] = 0;
			for (int k = 0; k < l->size; k++) {
				l1->neurons[j] += l->neurons[k] * l->weights[k][j];
			}
			l1->neurons[j] += l1->biases[j];
			l1->neurons[j] = sigmoid(l1->neurons[j]);
		}
	}

	memcpy(outputs, nnlayers[layers - 1]->neurons, nnlayers[layers - 1]->size * sizeof(double));
}

void NeuralNetwork::Backpropogation(double* targets) {
    double* errors = new double[nnlayers[layers - 1]->size];
    for (int i = 0; i < nnlayers[layers - 1]->size; i++) {
        errors[i] = targets[i] - nnlayers[layers - 1]->neurons[i];
    }

    for (int k = layers - 2; k >= 0; k--) {
        NNLayer* l = nnlayers[k];
        NNLayer* l1 = nnlayers[k + 1];
        double* errorsNext = new double[l->size];
        double* gradients = new double[l1->size];

        for (int i = 0; i < l1->size; i++) {
            gradients[i] = errors[i] * dsigmoid(nnlayers[k + 1]->neurons[i]);
            gradients[i] *= learningRate;
        }

        double** deltas = (double**)calloc(l1->size, sizeof(double*));
        for (int i = 0; i < l1->size; i++) deltas[i] = (double*)calloc(l->size, sizeof(double));

        for (int i = 0; i < l1->size; i++) {
            for (int j = 0; j < l->size; j++) {
                deltas[i][j] = gradients[i] * l->neurons[j];
            }
        }

        for (int i = 0; i < l->size; i++) {
            errorsNext[i] = 0;
            for (int j = 0; j < l1->size; j++) {
                errorsNext[i] += l->weights[i][j] * errors[j];
            }
        }

        free(errors);
        errors = errorsNext;

        for (int i = 0; i < l1->size; i++) {
            for (int j = 0; j < l->size; j++) {
                l->weights[j][i] += deltas[i][j];
            }
        }

        for (int i = 0; i < l1->size; i++) {
            l1->biases[i] += gradients[i];
        }

        free(gradients);
        for (int i = 0; i < l1->size; i++) free(deltas[i]);
        free(deltas);
    }
    free(errors);
}