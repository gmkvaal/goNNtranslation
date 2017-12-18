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
}

// setHyperParameters initiates the hyper parameters
func (hp *hyperParameters) initHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed updates all neurons for input x
func (nf *NetworkFormat) forwardFeed(x []float64, l int) []float64 {
	nf.activations[0] = x
	for k := 0; k < l; k++ {
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
func (nf *NetworkFormat) outputError(y []float64, l int) {
	for j := 0; j < nf.Sizes[l]; j++ {
		nf.delta[l-1][j] = outputNeuronError(nf.z[l-1][j], nf.activations[l][j], y[j])
	}
}

func (nf *NetworkFormat) outputGradients(nablaW [][][]float64, nablaB [][]float64, l int) ([][]float64, []float64) {
	for j := 0; j < nf.Sizes[l]; j++ {
		nablaB[l-1][j] += nf.delta[l-1][j]
		for i := 0; i < nf.Sizes[l-1]; i++ {
			nablaW[l-1][j][i] += nf.delta[l-1][j]*nf.activations[l-1][i]
		}
	}

	return nablaW[l-1], nablaB[l-1]
}

// backPropError backpropagates the error through the hidden layers
func (nf *NetworkFormat) backPropError(nablaW [][][]float64, nablaB [][]float64, l int) ([][][]float64, [][]float64){
	for k := 2; k < l+1; k++ {
		for j := 0; j < nf.Sizes[l+1-k]; j++ {
			nf.delta[l-k][j] = 0
			for i := 0; i < nf.Sizes[l+2-k]; i++ {
				nf.delta[l-k][j] += nf.weights[l+1-k][i][j] * nf.delta[l+1-k][i] * sigmoidPrime(nf.z[l-k][j])
			}
			nablaB[l-k][j] +=  nf.delta[l-k][j]

			for i := 0; i < nf.Sizes[l-k]; i++ {
				nablaW[l-k][j][i] += nf.delta[l-k][j] * nf.activations[l-k][i]
			}
		}
	}

	return nablaW, nablaB
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (nf *NetworkFormat) backPropAlgorithm(x, y []float64) ([][][]float64, [][]float64){

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	l := len(nf.Sizes) - 1 // last entry "layer-vise"

	// 1. Forward feed
	nf.forwardFeed(x, l)

	// 2. Computing the output error (delta L).
	nf.outputError(y, l)

	// 3. Gradients at the output layer
	nablaW[l-1], nablaB[l-1] = nf.outputGradients(nablaW, nablaB, l)

	// 4. Backpropagating the error
	nablaW, nablaB = nf.backPropError(nablaW, nablaB, l)

	return nablaW, nablaB
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (nf *NetworkFormat) updateMiniBatches() {
	deltaNablaW := nf.cubicMatrix(zeroFunc())
	deltaNablaB := nf.squareMatrix(zeroFunc())

	for i := range nf.data.miniBatches {
		nablaW := nf.cubicMatrix(zeroFunc())
		nablaB := nf.squareMatrix(zeroFunc())

		for _, dataSet := range nf.data.miniBatches[i] {
			deltaNablaW, deltaNablaB = nf.backPropAlgorithm(dataSet[0], dataSet[1])
			nablaW = nf.updateNablaW(deltaNablaW, nablaW)
			nablaB = nf.updateNablaB(deltaNablaB, nablaB)
		}

		nf.updateWeights(nablaW)
		nf.updateBiases(nablaB)
	}
}


func (nf *NetworkFormat) updateNablaB(deltaNablaB [][]float64,	nablaB [][]float64) [][]float64 {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
				nablaB[k][j] += deltaNablaB[k][j]
				}
	}

	return nablaB
}

func (nf *NetworkFormat) updateNablaW(deltaNablaW[][][]float64,	nablaW [][][]float64) [][][]float64 {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			for i := 0; i < nf.Sizes[k]; i++ {
				nablaW[k][j][i] += deltaNablaW[k][j][i]
			}
		}
	}

	return nablaW
}

// updateWeights updates the weight matrix following a mini batch
func (nf *NetworkFormat) updateWeights(nablaW [][][]float64) {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			for i := 0; i < nf.Sizes[k]; i++ {
				nf.weights[k][j][i] = (1 - nf.hp.eta*(nf.hp.lambda/nf.data.n))*nf.weights[k][j][i] -
					nf.hp.eta/nf.data.miniBatchSize * nablaW[k][j][i]
			}
		}
	}
}

// updateBiases updates the bias matrix following a mini batch
func (nf *NetworkFormat) updateBiases(nablaB [][]float64) {
	for k := 0; k < len(nf.Sizes) - 1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			nf.biases[k][j] = nf.biases[k][j] - nf.hp.eta/nf.data.miniBatchSize*nablaB[k][j]
		}
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

/*

func main() {

	nf := NetworkFormat{Sizes: []int{784, 30, 10}}
	nf.trainNetwork(1000,15, 5, 5, 0.1, true)

	fmt.Println("")


}

*/