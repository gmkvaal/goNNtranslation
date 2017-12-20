package main

import (

	NN "github.com/gmkvaal/goNNtranslation"


)


func main() {





	n := NN.Network{}
	n.AddLayer(784, NN.Sigmoid, NN.SigmoidPrime)
	n.AddLayer(30, NN.Sigmoid, NN.SigmoidPrime)
	n.AddLayer(10, NN.Sigmoid, NN.SigmoidPrime)

	n.InitNetworkMethods(NN.OutputErrorXEntropy)

	n.TrainNetwork(1000, 30, 10, 0.5, 5, true)
}
