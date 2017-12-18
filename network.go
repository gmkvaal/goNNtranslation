package network

import (
	"fmt"
)

// NetworkFormat contains the
// fields Sizes, biases, and weights
type NetworkFormat struct {
	Sizes       []int
	biases      [][]float64
	weights     [][][]float64
	delta       [][]float64
	nablaW      [][][]float64
	nablaB      [][]float64
	z           [][]float64
	activations [][]float64
	l 			int
	data
	hp hyperParameters
}

type hyperParameters struct {
	eta float64
	lambda float64
}

// initNetwork initiates the weights
// and biases with random numbers
func (nf *NetworkFormat) initNetwork() {
	nf.weights = nf.cubicMatrix(randomFunc())
	nf.biases = nf.squareMatrix(randomFunc())
	nf.delta = nf.squareMatrix(zeroFunc())
	nf.nablaW = nf.cubicMatrix(zeroFunc())
	nf.nablaB = nf.squareMatrix(zeroFunc())
	nf.z = nf.squareMatrix(zeroFunc())
	nf.activations = nf.squareMatrixFull(zeroFunc())
	nf.l = len(nf.Sizes) - 1
}

// setHyperParameters initiates the hyper parameters
func (hp *hyperParameters) initHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed updates all neurons for input x
func (nf *NetworkFormat) forwardFeed(x []float64) []float64 {
	nf.activations[0] = x
	for k := 0; k < nf.l; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			sum := 0.0
			for i := 0; i < nf.Sizes[k]; i++ {
				sum += nf.activations[k][i] * nf.weights[k][j][i]
			}
			nf.z[k][j] = sum + nf.biases[k][j]
			nf.activations[k+1][j] = sigmoid(nf.z[k][j])
		}
	}

	return nf.activations[2]
}

// outputError computes the error at the output neurons
func (nf *NetworkFormat) outputError(y []float64) {
	for j := 0; j < nf.Sizes[nf.l]; j++ {
		nf.delta[nf.l-1][j] = outputNeuronError(nf.z[nf.l-1][j], nf.activations[nf.l][j], y[j])
	}
}

func (nf *NetworkFormat) outputGradients()  {
	for j := 0; j < nf.Sizes[nf.l]; j++ {
		nf.nablaB[nf.l-1][j] += nf.delta[nf.l-1][j]
		for i := 0; i < nf.Sizes[nf.l-1]; i++ {
			nf.nablaW[nf.l-1][j][i] += nf.delta[nf.l-1][j]*nf.activations[nf.l-1][i]
		}
	}
}

// backPropError backpropagates the error through the hidden layers
func (nf *NetworkFormat) backPropError() {
	for k := 2; k < nf.l+1; k++ {
		for j := 0; j < nf.Sizes[nf.l+1-k]; j++ {
			nf.delta[nf.l-k][j] = 0
			for i := 0; i < nf.Sizes[nf.l+2-k]; i++ {
				nf.delta[nf.l-k][j] += nf.weights[nf.l+1-k][i][j] * nf.delta[nf.l+1-k][i] *
					sigmoidPrime(nf.z[nf.l-k][j])
			}
			nf.nablaB[nf.l-k][j] +=  nf.delta[nf.l-k][j]

			for i := 0; i < nf.Sizes[nf.l-k]; i++ {
				nf.nablaW[nf.l-k][j][i] += nf.delta[nf.l-k][j] * nf.activations[nf.l-k][i]
			}
		}
	}
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (nf *NetworkFormat) backPropAlgorithm(x, y []float64) {

	// 1. Forward feed
	nf.forwardFeed(x)

	// 2. Computing the output error (delta L).
	nf.outputError(y)

	// 3. Gradients at the output layer
	nf.outputGradients()

	// 4. Backpropagating the error
	nf.backPropError()
}

// updateWeights updates the weight matrix following a mini batch
func (nf *NetworkFormat) updateWeights() {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			for i := 0; i < nf.Sizes[k]; i++ {
				nf.weights[k][j][i] = (1 - nf.hp.eta*(nf.hp.lambda/nf.data.n))*nf.weights[k][j][i] -
					nf.hp.eta/nf.data.miniBatchSize * nf.nablaW[k][j][i]
			}
		}
	}
}

// updateBiases updates the bias matrix following a mini batch
func (nf *NetworkFormat) updateBiases() {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			nf.biases[k][j] = nf.biases[k][j] - nf.hp.eta/nf.data.miniBatchSize*nf.nablaB[k][j]
		}
	}
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (nf *NetworkFormat) updateMiniBatches() {
	for i := range nf.data.miniBatches {
		nf.nablaW = nf.cubicMatrix(zeroFunc())
		nf.nablaB = nf.squareMatrix(zeroFunc())

		for _, dataSet := range nf.data.miniBatches[i] {
			nf.backPropAlgorithm(dataSet[0], dataSet[1])
		}

		nf.updateWeights()
		nf.updateBiases()
	}
}

// trainNetwork trains the network with the parameters given as arguments
func (nf *NetworkFormat) TrainNetwork(dataCap int, epochs int, miniBatchSize int, eta, lambda float64, shuffle bool) {
	nf.initNetwork()
	nf.data.loadData()
	nf.hp.initHyperParameters(eta, lambda)

	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch", i, ":")

		nf.data.miniBatchGenerator(0, dataCap, miniBatchSize, shuffle)
		nf.updateMiniBatches()
		nf.validate(nf.data.validationInput, nf.data.validationOutput, 1000)
		//nf.validate(nf.data.trainingInput, nf.data.trainingOutput, 100)

		fmt.Println("Avg cost:", nf.totalCost(nf.data.validationInput[:dataCap], nf.data.validationInput[:dataCap]))
		fmt.Println("")
	}
}
