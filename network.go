package main

import (
	"fmt"
	"time"
	"log"
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
	hyperParameters
}

type hyperParameters struct {
	eta float64
	lambda float64
}

func (hp *hyperParameters) setHyperParameters (eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
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
func (nf *networkFormat) backProp(x []float64, y []float64, nablaW[][][]float64, nablaB[][]float64) ([][][]float64, [][]float64){
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
	nablaB[l-1] = nf.delta[l-1]
	nablaW[l-1] = vectorMatrixProduct(nablaW[l-1], nf.delta[l-1], nf.activations[l-1])

	// Backpropagating the error
	for k := 2; k < l+1; k++ {
		for j := 0; j < nf.sizes[l+1-k]; j++ {
			sum = 0
			for i := 0; i < nf.sizes[l+2-k]; i++ {
				sum += nf.weights[l+1-k][i][j] * nf.delta[l+1-k][i] * sigmoidPrime(nf.z[l+1-k][i])
			}
			nf.delta[l-k][j] = sum
			nablaB[l-k][j] +=  nf.delta[l-k][j]

			for i := 0; i < nf.sizes[l-k]; i++ {
				nablaW[l-k][j][i] += nf.delta[l-k][j] * nf.activations[l-k][i]
			}
		}
	}

	return nablaW, nablaB
}

func (nf *networkFormat) updateWeights() {
	for k := 0; k < len(nf.sizes) - 2; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			for i := 0; i < nf.sizes[k]; i++ {
				//fmt.Println("k", k, "j", j, "i", i)
				//fmt.Println(len(nf.weights[k][j]))
				nf.weights[k][j][i] = (1 - nf.hyperParameters.eta*(nf.hyperParameters.lambda/float64(nf.data.n)))*nf.weights[k][j][i] -
					nf.hyperParameters.eta/float64(nf.data.miniBatchSize) * nf.nablaW[k][j][i]
				nf.nablaW[k][j][i] = 0.0 // clearing for next batch
			}
		}
	}
}

func (nf *networkFormat) updateBiases() {
	for k := 0; k < len(nf.sizes) - 2; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			nf.biases[k][j] = nf.biases[k][j] - nf.hyperParameters.eta/float64(nf.data.miniBatchSize) * nf.nablaB[k][j]
			nf.nablaB[k][j] = 0
		}
	}
}

func (nf *networkFormat) updateMiniBatch(eta float64, lambda float64, n int, miniBatchSize int) {
	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	for i := range nf.data.miniBatches {
		//fmt.Println(idx, len(miniBatch))
		fmt.Println(i)
		for _, dataSet := range nf.data.miniBatches[i] {
			x := dataSet[0]
			y := dataSet[1]
			nablaW, nablaB = nf.backProp(x, y, nablaW, nablaB)
		}
		nf.updateWeights()
		nf.updateBiases()
	}
}

func (nf *networkFormat) trainNetwork(epochs int, miniBatchSize int, eta float64, lambda float64) {

	nf.data.formatData()
	nf.hyperParameters.setHyperParameters(eta, lambda)
	nf.data.miniBatchGenerator(5)
	nf.updateMiniBatch(1.0, 1.0, 10, miniBatchSize)
}

func main() {

	nf := &networkFormat{sizes: []int{784, 30, 10}}
	nf.initNetwork()

	//nf.backProp(x, y)

	nf.trainNetwork(1, 5, 0.5, 0.1)


	//6000, 10, 2, 784 / 10
	fmt.Println("")

	start := time.Now()
	nf.cubicMatrix(zeroFunc())
	elapsed := time.Since(start)
	log.Printf("Binomial took %s", elapsed)

	fmt.Println(nf.data.miniBatchSize)


}
