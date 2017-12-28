package network

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"log"
	"time"
	"runtime"
	"regexp"
)



// Network contains the
// fields Sizes, biases, and weights
type Network struct {
	Sizes       []int
	layer	layer
	layers []layer
	l           int
	hp          HyperParameters
	data
	NetworkMethods
	dataContainers
}

type layer struct {
	size int
	activationFunction
}

type dataContainers struct {
	weights     []*mat64.Dense
	biases      []*mat64.Dense
	nablaW      []*mat64.Dense
	nablaB      []*mat64.Dense
	deltaNablaW []*mat64.Dense
	deltaNablaB []*mat64.Dense
	delta       []*mat64.Dense
	z           []*mat64.Dense
	activations []*mat64.Dense
	sp          []*mat64.Dense
}

type activationFunction struct {
	function func(i, j int, v float64) float64
	prime func(i, j int, v float64) float64
}

type NetworkMethods struct {
	outputErrorFunc  func(delta *mat64.Dense, a, y mat64.Matrix)
	validationMethod func(n *Network, inputData, outputData []*mat64.Dense) bool
}

type HyperParameters struct {
	eta    float64
	lambda float64
}

func (n *Network) AddLayer(layerSize int, activationFunction, activationPrime func(i, j int, v float64) float64) {
	n.layer.size = layerSize
	n.layer.function = activationFunction
	n.layer.prime = activationPrime

	n.layers = append(n.layers, n.layer)
}


func (n *Network) setSizes(){
	for idx := range n.layers {
		n.Sizes = append(n.Sizes, n.layers[idx].size)
	}
}

// initNetwork initiates the weights
// and biases with random numbers
func (n *Network) initDataContainers() {
	n.setSizes()
	n.weights = n.cubicMatrix(randomFunc())
	n.biases = n.squareMatrix(randomFunc())
	n.delta = n.squareMatrix(zeroFunc())
	n.nablaW = n.cubicMatrix(zeroFunc())
	n.nablaB = n.squareMatrix(zeroFunc())
	n.z = n.squareMatrix(zeroFunc())
	n.activations = n.squareMatrixFull(zeroFunc())
	n.l 			= len(n.Sizes) - 1
}

func (nm *NetworkMethods) InitNetworkMethods(outputError func(delta *mat64.Dense, a, y mat64.Matrix),
	validationMethod func(n *Network, inputData, outputData []*mat64.Dense) bool) {
		nm.outputErrorFunc = outputError
		nm.validationMethod = validationMethod
}

// setHyperParameters initiates the hyper parameters
func (hp *HyperParameters) InitHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed updates all neurons for input x
func (n *Network) forwardFeed(x []float64) []float64 {
	n.activations[0] = x
	for k := 0; k < n.l; k++ {
		for j := 0; j < n.Sizes[k+1]; j++ {
			sum := 0.0
			for i := 0; i < n.Sizes[k]; i++ {
				sum += n.activations[k][i] * n.weights[k][j][i]
			}
			n.z[k][j] = sum + n.biases[k][j]
			n.activations[k+1][j] = sigmoid(n.z[k][j])
		}
	}

	return n.activations[2]
}

// outputError computes the error at the output neurons
func (n *Network) outputError(y []float64) {
	for j := 0; j < n.Sizes[n.l]; j++ {
		n.delta[n.l-1][j] = n.outputErrorFunc(n.z[n.l-1][j], n.activations[n.l][j], y[j])
	}
}

func (n *Network) outputGradients()  {
	for j := 0; j < n.Sizes[n.l]; j++ {
		n.nablaB[n.l-1][j] += n.delta[n.l-1][j]
		for i := 0; i < n.Sizes[n.l-1]; i++ {
			n.nablaW[n.l-1][j][i] += n.delta[n.l-1][j]*n.activations[n.l-1][i]
		}
	}
}

// backPropError backpropagates the error through the hidden layers
func (n *Network) backPropError() {
	for k := 2; k < n.l+1; k++ {
		for j := 0; j < n.Sizes[n.l+1-k]; j++ {
			n.delta[n.l-k][j] = 0
			for i := 0; i < n.Sizes[n.l+2-k]; i++ {
				n.delta[n.l-k][j] += n.weights[n.l+1-k][i][j] * n.delta[n.l+1-k][i] *
					sigmoidPrime(n.z[n.l-k][j])
			}
			n.nablaB[n.l-k][j] +=  n.delta[n.l-k][j]

			for i := 0; i < n.Sizes[n.l-k]; i++ {
				n.nablaW[n.l-k][j][i] += n.delta[n.l-k][j] * n.activations[n.l-k][i]
			}
		}
	}
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (n *Network) BackPropAlgorithm(x, y *mat64.Dense) {
	//defer TimeTrack(time.Now())

	// 1. Forward feed
	n.forwardFeed(x)

	// 2. Computing the output error (delta L).
	n.outputError(y)

	// 3. Gradients at the output layer
	n.outputGradients()

	// 4. Backpropagating the error
	n.backPropError()
}

// updateWeights updates the weight matrix following a mini batch
func (n *Network) updateWeights() {
	for k := 0; k < len(n.Sizes) - 1; k++ {
		for j := 0; j < n.Sizes[k+1]; j++ {
			for i := 0; i < n.Sizes[k]; i++ {
				n.weights[k][j][i] = (1 - n.hp.eta*(n.hp.lambda/n.data.n))*n.weights[k][j][i] -
					n.hp.eta/n.data.miniBatchSize * n.nablaW[k][j][i]
			}
		}
	}
}

// updateBiases updates the bias matrix following a mini batch
func (n *Network) updateBiases() {
	for k := 0; k < len(n.Sizes) - 1; k++ {
		for j := 0; j < n.Sizes[k+1]; j++ {
			n.biases[k][j] = n.biases[k][j] - n.hp.eta/n.data.miniBatchSize*n.nablaB[k][j]
		}
	}
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (n *Network) updateMiniBatches() {
	for i := range n.data.miniBatches {
		n.nablaW = n.cubicMatrix(zeroFunc())
		n.nablaB = n.squareMatrix(zeroFunc())

		for _, dataSet := range n.data.miniBatches[i] {
			n.backPropAlgorithm(dataSet[0], dataSet[1])
		}

		n.updateWeights()
		n.updateBiases()
	}
}


// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (n *Network) updateMiniBatches() {
	//defer TimeTrack(time.Now())

	for i := range n.data.miniBatches {
		for _, dataSet := range n.data.miniBatches[i] {
			n.BackPropAlgorithm(dataSet[0], dataSet[1])
			n.updateGradients()
		}

		n.updateWeightsAndBiases()
	}
}

// trainNetwork trains the network with the parameters given as arguments
func (n *Network) TrainNetwork(epochs int, miniBatchSize int, eta, lambda float64, shuffle, validate bool) {
	defer TimeTrack(time.Now())


	if len(n.trainingInput) == 0 || len(n.trainingOutput) == 0 {
		log.Fatal("Insufficient training data submitted")
	}

	if validate {
		if len(n.validationInput) == 0 || len(n.validationOutput) == 0 {
			log.Fatal("Insufficient validation data submitted")
		}
	}

	n.initDataContainers()
	n.hp.InitHyperParameters(eta, lambda)

	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch", i, ":")

		n.data.miniBatchGenerator(miniBatchSize, shuffle)
		n.updateMiniBatches()

		if validate {
			n.validationMethod(n, n.data.validationInput, n.data.validationOutput)
		}

		//fmt.Println("Avg cost:", nf.totalCost(nf.data.validationInput[:dataCap], nf.data.validationInput[:dataCap]))
		fmt.Println("")
	}
}

func TimeTrack(start time.Time) {
	elapsed := time.Since(start)

	// Skip this function, and fetch the PC and file for its parent.
	pc, _, _, _ := runtime.Caller(1)

	// Retrieve a function object this functions parent.
	funcObj := runtime.FuncForPC(pc)

	// Regex to extract just the function name (and not the module path).
	runtimeFunc := regexp.MustCompile(`^.*\.(.*)$`)
	name := runtimeFunc.ReplaceAllString(funcObj.Name(), "$1")

	log.Println(fmt.Sprintf("%s took %s", name, elapsed))
}