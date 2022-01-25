#include "NNSaver.h"

void NNSaver::Save(NeuralNetwork* nn, string path)
{
	ofstream f;
	f.open(path.c_str(), ios::binary);
	f.write((char*)&nn->layers, 4);
	f.write((char*)&nn->learningRate, 8);

	int* sizes = new int[nn->layers];
	for (int i = 0; i < nn->layers; i++) sizes[i] = nn->nnlayers[i]->size;
	f.write((char*)sizes, nn->layers * 4);
	free(sizes);

	for (int i = 0; i < nn->layers; i++) {
		NNLayer* layer = nn->nnlayers[i];
		int layer1size = (i < nn->layers - 1) ? (nn->nnlayers[i + 1]->size) : 0;
		f.write((char*)&layer->size, 4);
		f.write((char*)&layer1size, 4);
		f.write((char*)layer->biases, layer->size * 8);
		for (int j = 0; j < layer->size; j++) {
			f.write((char*)layer->weights[j], layer1size * 8);
		}
	}

	f.close();
}

NeuralNetwork* NNSaver::Load(string path)
{
	ifstream f;
	f.open(path.c_str(), ios::binary);

	int layers;
	double learningRate;

	f.read((char*)&layers, 4);
	f.read((char*)&learningRate, 8);

	int* sizes = new int[layers];
	f.read((char*)sizes, layers * 4);
	NeuralNetwork* nn = new NeuralNetwork(learningRate, layers, sizes);
	free(sizes);

	for (int i = 0; i < layers; i++) {
		NNLayer* layer = nn->nnlayers[i];
		int layersize, layer1size;
		f.read((char*)&layersize, 4);
		f.read((char*)&layer1size, 4);
		f.read((char*)layer->biases, layersize * 8);
		for (int j = 0; j < layersize; j++) {
			f.read((char*)layer->weights[j], layer1size * 8);
		}
	}

	f.close();

	return nn;
}
