package main

import (

	NN "github.com/gmkvaal/goNNtranslation"


)


func main() {





	nf := NN.NetworkFormat{Sizes: []int{784, 30, 10}}
	nf.InitNetworkMethods(NN.OutputErrorXEntropy, NN.SigmoidActivation, NN.SigmoidActivationPrime)
	nf.TrainNetwork(6000, 30, 10, 0.5, 5, true)
}
