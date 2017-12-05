package main

import (
	"testing"

	"fmt"
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


func TestOutputGradients(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.sizes) - 1

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	nf.outputGradients(nablaW, nablaB, l)
}
*/

func TestBackProp(t *testing.T) {

	// 2. activations ok (forward feed ok)
	// 3. delta ok
	// 4. gradients at output ok
	// 5. nablab not ok

	nf := networkFormat{sizes: []int{784, 30, 10}}
	nf.initNetwork()
	//	nf.trainNetwork(1000,1, 10, 1, 2, false)

	nf.miniBatchSize = 1
	nf.n = 1
	nf.hp.eta = 1
	nf.hp.lambda = 5.0

	x := make([]float64, 784, 784)
	y := make([]float64, 10, 10)

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	nablaW, nablaB = nf.backPropAlgorithm(x, y, nablaW, nablaB)

	nf.updateBiases(nablaB)
	nf.updateWeights(nablaW)

	//fmt.Println(nf.biases[1])
	//fmt.Println(nablaB[1])

	//x[2] = 0.53
	//x[3] = 0.19
	//x[4] = 0.23
	//x[78] = 0.59
	//x[394] = 0.23
	y[0] = 1
	x[1] = 1


	nablaW = nf.cubicMatrix(zeroFunc())
	nablaB = nf.squareMatrix(zeroFunc())

	fmt.Println("new backprop")

	nablaW, nablaB = nf.backPropAlgorithm(x, y, nablaW, nablaB)

	nf.updateBiases(nablaB)
	nf.updateWeights(nablaW)

	//fmt.Println(nablaW[0][0])
	//fmt.Println("new")
	//fmt.Println(nf.weights[0][0])



	//-0.49296818647837726
	//0.007031813521622738

}

