package main

import (

	NN "github.com/gmkvaal/goNNtranslation"

	"github.com/gmkvaal/goNNtranslation/train/MNIST"
)


func main() {

	n := NN.Network{}
	n.AddLayer(784, NN.Sigmoid, NN.SigmoidPrime)
	n.AddLayer(30, NN.Sigmoid, NN.SigmoidPrime)
	n.AddLayer(10, NN.Sigmoid, NN.SigmoidPrime)
	n.InitNetworkMethods(NN.OutputErrorXEntropy)



	data := MNIST.Data{}
	data.FormatMNISTData()

	n.LoadTrainingData(data.TrainingInput[:1000], data.TrainingOutput[:1000])
	n.LoadValidationData(data.ValidationInput[:100], data.ValidationOutput[:100])


	n.TrainNetwork(30, 10, 0.5, 5, true)
}
