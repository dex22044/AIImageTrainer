all:
	g++ -o app.bin -std=c++17 -O3 \
	main.cpp \
	NeuralNetwork.cpp \
	NNLayer.cpp \
	NNSaver.cpp \
	-lSDL2