package main

import (
	"testing"

)
/*

// TestForwardFeed tests the forward feed algorithm
// by initiating with zero-weights and biases, hence making
// all z's zero and thus all activations 1/2 (given sigmoids)
func TestForwardFeed(t *testing.T) {
	nf = initNetworkForTesting()
	nf.weights = nf.cubicMatrix(zeroFunc())
	nf.biases = nf.squareMatrix(zeroFunc())
	l := len(nf.sizes) - 1

	x := make([]float64, 784, 784)
	y := []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	assert.Equal(t, nf.forwardFeed(x, l), y)
}

func TestOutputError(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.sizes) - 1

	y := []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	yNeg := []float64{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	nf.outputError(y, l)
	assert.Equal(t, nf.delta[l-1], yNeg)
}

func TestBackProp(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.sizes) - 1

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())
	nf.delta[l-1] = []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	nablaB[l-1] = nf.delta[l-1]
	nablaW[l-1] = vectorMatrixProduct(nablaW[l-1], nf.delta[l-1], nf.activations[l-1])
	nablaW, nablaB = nf.backPropError(nablaW, nablaB, l)
	nablaB[l-1] = nf.delta[l-1]
	nablaW[l-1] = vectorMatrixProduct(nablaW[l-1], nf.delta[l-1], nf.activations[l-1])
	nablaW, nablaB = nf.backPropError(nablaW, nablaB, l)

	fmt.Println("nab")
	fmt.Println(nablaB)
}

*/
func TestOutputGradients(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.sizes) - 1

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	nf.outputGradients(nablaW, nablaB, l)
}