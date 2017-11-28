package main

import (
	"fmt"
)


// networkFormat contains the
// fields sizes, biases, and weights
type networkFormat struct {
	sizes       []int
	biases      [][]float64
	weights     [][][]float64
	delta       [][]float64
	nablaW      [][][]float64
	nablaB      [][]float64
	z           [][]float64
	activations [][]float64
	data
}

// initNetwork initiates the weights
// and biases with random numbers
func (nf *networkFormat) initNetwork() {
	nf.weights = nf.cubicMatrix(randomFunc())
	nf.biases = nf.squareMatrix(randomFunc())
	nf.delta = nf.squareMatrix(zeroFunc())
	nf.nablaW = nf.cubicMatrix(zeroFunc())
	nf.nablaB = nf.squareMatrix(zeroFunc())
	nf.z = nf.squareMatrix(zeroFunc())
	nf.activations = nf.squareMatrixFull(zeroFunc())
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (nf *networkFormat) backProp(x []float64, y []float64) {

	fmt.Println(nf.nablaW[1][0])

	var sum float64 = 0
	l := len(nf.sizes) - 1 // last entry "layer-vise"

	// Clearing / preparing the slices
	nf.activations[0] = x

	// Updating all neurons with the forwardFeed algorithm
	for k := 0; k < l; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			nf.z[k][j] = dot(nf.activations[k], nf.weights[k][j]) + nf.biases[k][j]
			nf.activations[k+1][j] = sigmoid(nf.z[k][j])
		}
	}

	// Computing the output error (delta L).
	for j := 0; j < nf.sizes[l]; j++ {
		nf.delta[l-1][j] = outputNeuronError(nf.z[l-1][j], nf.activations[l][j], y[j])
	}

	// Gradients at the output layer
	nf.nablaB[l-1] = nf.delta[l-1]
	nf.nablaW[l-1] = vectorMatrixProduct(nf.nablaW[l-1], nf.delta[l-1], nf.activations[l-1])

	// Backpropagating the error
	for k := 2; k < l+1; k++ {
		for j := 0; j < nf.sizes[l+1-k]; j++ {
			sum = 0
			for i := 0; i < nf.sizes[l+2-k]; i++ {
				sum += nf.weights[l+1-k][i][j] * nf.delta[l+1-k][i] * sigmoidPrime(nf.z[l+1-k][i])
			}
			nf.delta[l-k][j] = sum
			nf.nablaB[l-k][j] +=  nf.delta[l-k][j]

			for i := 0; i < nf.sizes[l-k]; i++ {
				nf.nablaW[l-k][j][i] += nf.delta[l-k][j] * nf.activations[l-k][i]
			}
		}
	}
}

func (nf *networkFormat) updateWeights(eta float64, lambda float64, n int, miniBatchSize int) {
	for k := 0; k < len(nf.sizes) - 2; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			for i := 0; i < nf.sizes[k]; i++ {
				//fmt.Println("k", k, "j", j, "i", i)
				//fmt.Println(len(nf.weights[k][j]))
				nf.weights[k][j][i] = (1 - eta*(lambda/float64(n)))*nf.weights[k][j][i] -
					eta/float64(miniBatchSize) * nf.nablaW[k][j][i]
				nf.nablaW[k][j][i] = 0.0 // clearing for next batch
			}
		}
	}
}

func (nf *networkFormat) updateBiases(eta float64, lambda float64, n int, miniBatchSize int) {
	for k := 0; k < len(nf.sizes) - 2; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			nf.biases[k][j] = nf.biases[k][j] - eta/float64(miniBatchSize) * nf.nablaB[k][j]
			nf.nablaB[k][j] = 0
		}
	}
}

func (nf *networkFormat) updateMiniBatch(eta float64, lambda float64, n int, miniBatchSize int) {
	for i := range nf.data.miniBatches {
		//fmt.Println(idx, len(miniBatch))
		for _, dataSet := range nf.data.miniBatches[i] {
			x := dataSet[0]
			y := dataSet[1]
			nf.backProp(x, y)
		}
		nf.updateWeights(eta, lambda, n, miniBatchSize)
		nf.updateBiases(eta, lambda, n, miniBatchSize)
	}
}

func (nf *networkFormat) trainNetwork(epochs int, miniBatchSize int, lambda float64) {

	nf.data.formatData()
	nf.data.miniBatchGenerator(10)
	nf.updateMiniBatch(1.0, 1.0, 6000, miniBatchSize)
}

func main() {

	nf := &networkFormat{sizes: []int{784, 30, 10}}
	nf.initNetwork()

	//nf.backProp(x, y)

	nf.trainNetwork(1, 5, 0.5)


	//6000, 10, 2, 784 / 10
	fmt.Println("")


}
