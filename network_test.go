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
/*
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

	fmt.Println(nablaB[:])
	//fmt.Println("new")
	//fmt.Println(nf.weights[0][0])
}
*/

func TestBackProp2(t *testing.T) {
	nf := networkFormat{sizes: []int{784, 30, 10}}

	nf.weights = nf.cubicMatrix(zeroFunc())
	nf.biases = nf.squareMatrix(zeroFunc())
	nf.delta = nf.squareMatrix(zeroFunc())
	//nf.nablaW = nf.cubicMatrix(zeroFunc())
	//nf.nablaB = nf.squareMatrix(zeroFunc())
	nf.z = nf.squareMatrix(zeroFunc())
	nf.activations = nf.squareMatrixFull(zeroFunc())


	nf.miniBatchSize = 2
	nf.n = 4
	nf.hp.eta = 1
	nf.hp.lambda = 5.0

	x1 := make([]float64, 784, 784)
	y1 := make([]float64, 10, 10)
	y1[0] = 1
	x1[1] = 1
	b1 := [][]float64{x1, y1}

	x2 := make([]float64, 784, 784)
	y2 := make([]float64, 10, 10)
	y2[1] = 1
	x2[2] = 1
	b2 := [][]float64{x2, y2}

	miniBatchA := [][][]float64{b1, b2}

	x3 := make([]float64, 784, 784)
	y3 := make([]float64, 10, 10)
	y3[2] = 1
	x3[3] = 1
	b3 := [][]float64{x3, y3}

	x4 := make([]float64, 784, 784)
	y4 := make([]float64, 10, 10)
	y4[3] = 1
	x4[4] = 1
	b4 := [][]float64{x4, y4}

	miniBatchB := [][][]float64{b3, b4}

	miniBatches := [][][][]float64{miniBatchA, miniBatchB}

	nf.data.miniBatches = miniBatches

	nf.updateMiniBatches()

	fmt.Println(nf.biases)

}
