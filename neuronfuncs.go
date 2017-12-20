package network

import (
	"math"
	"github.com/gonum/matrix/mat64"
)


// OutputErrorXEntropy computes `delta` at the output layer
// for output activations `a` and output matrix `y`
func OutputErrorXEntropy(delta *mat64.Dense, a, y mat64.Matrix) {
	delta.Sub(a, y)
}

func Sigmoid(i, j int, v float64) float64 {
	return sigmoid(v)
}

// sigmoid returns the sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func SigmoidPrime(i, j int, v float64) float64 {
	return sigmoidPrime(v)
}

// sigmoidPrime returns the differentiated sigmoid function
func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

