package network

import (
	"math"
)



// sigmoid returns the sigmoid function
func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// sigmoidPrime returns the differentiated sigmoid function
func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}

func OutputNeuronError(z float64, a float64, y float64) float64 {
	return a - y //* sigmoidPrime(z)
}