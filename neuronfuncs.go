package main

import "math"

// sigmoid returns the sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// sigmoidPrime returns the differentiated sigmoid function
func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

// delta returns the error at a given neuron
func outputNeuronError(z float64, a float64, y float64) float64 {
	return (a - y) * sigmoidPrime(z)
}
