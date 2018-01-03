package network

import (
	"fmt"
	"log"
	"math"
	"regexp"
	"runtime"
	"sync"
	"time"
)

var wg sync.WaitGroup

// Network contains the
// fields Sizes, biases, and weights
type Network struct {
	Sizes  []int
	layer  layer
	layers []layer
	l      int
	nCores int
	hp     HyperParameters
	data
	NetworkMethods
	dataContainers
}

type layer struct {
	size int
	activationFunction
	intervals [][]int
}

type dataContainers struct {
	Sizes       []int
	biases      [][]float64
	weights     [][][]float64
	delta       [][][]float64
	nablaW      [][][][]float64
	nablaB      [][][]float64
	z           [][][]float64
	activations [][][]float64
}

type activationFunction struct {
	function func(v float64) float64
	prime    func(v float64) float64
}

type NetworkMethods struct {
	outputErrorFunc  func(z float64, a float64, y float64) float64
	validationMethod func(n *Network, inputData, outputData [][]float64) bool
}

type HyperParameters struct {
	eta    float64
	lambda float64
}

func (n *Network) AddLayer(layerSize int, activationFunction, activationPrime func(v float64) float64,
	intervals []int) {
	n.layer.size = layerSize
	n.layer.function = activationFunction
	n.layer.prime = activationPrime

	intervalStart := 0
	for _, intervalStop := range intervals {
		n.layer.intervals = append(n.layer.intervals, []int{intervalStart, intervalStop+intervalStart})
		intervalStart += intervalStop
	}

	n.layers = append(n.layers, n.layer)
	n.layer.intervals = nil
}

func (n *Network) setSizes() {
	for idx := range n.layers {
		n.Sizes = append(n.Sizes, n.layers[idx].size)
	}
}

// initNetwork initiates the weights
// and biases with random numbers
func (n *Network) initDataContainers(nCores int) {
	n.setSizes()
	n.l = len(n.Sizes) - 1
	n.nCores = nCores
	n.weights = n.cubicMatrix(randomFunc())
	n.biases = n.squareMatrix(randomFunc())
	for idx := 0; idx < n.nCores; idx++ {
		n.delta = append(n.delta, n.squareMatrix(zeroFunc()))
		n.nablaW = append(n.nablaW, n.cubicMatrix(zeroFunc()))
		n.nablaB = append(n.nablaB, n.squareMatrix(zeroFunc()))
		n.z = append(n.z, n.squareMatrix(zeroFunc()))
		n.activations = append(n.activations, n.squareMatrixFull(zeroFunc()))
	}
}

func (nm *NetworkMethods) InitNetworkMethods(outputError func(z float64, a float64, y float64) float64,
	validationMethod func(n *Network, inputData, outputData [][]float64) bool) {
	nm.outputErrorFunc = outputError
	nm.validationMethod = validationMethod
}

// setHyperParameters initiates the hyper parameters
func (hp *HyperParameters) InitHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed updates all neurons for input x
func (n *Network) forwardFeed(x []float64, proc int) []float64 {
	n.activations[proc][0] = x
	for k := 0; k < n.l; k++ {
		for j := 0; j < n.Sizes[k+1]; j++ {
			sum := 0.0
			for i := 0; i < n.Sizes[k]; i++ {
				sum += n.activations[proc][k][i] * n.weights[k][j][i]
			}
			n.z[proc][k][j] = sum + n.biases[k][j]
			n.activations[proc][k+1][j] = n.layers[k+1].function(n.z[proc][k][j])
		}
	}

	return n.activations[proc][n.l]
}

// outputError computes the error at the output neurons
func (n *Network) outputError(y []float64, proc int) {
	for j := 0; j < n.Sizes[n.l]; j++ {
		n.delta[proc][n.l-1][j] = n.outputErrorFunc(n.z[proc][n.l-1][j], n.activations[proc][n.l][j], y[j])
	}
}

func (n *Network) outputGradients(proc int) {
	for j := 0; j < n.Sizes[n.l]; j++ {
		n.nablaB[proc][n.l-1][j] += n.delta[proc][n.l-1][j]
		for i := 0; i < n.Sizes[n.l-1]; i++ {
			n.nablaW[proc][n.l-1][j][i] += n.delta[proc][n.l-1][j] * n.activations[proc][n.l-1][i]
		}
	}
}

// backPropError backpropagates the error through the hidden layers
func (n *Network) backPropError(proc int) {
	for k := 2; k < n.l+1; k++ {
		go func(k int) {
			for j := 0; j < n.Sizes[n.l+1-k]; j++ {
				go func(j int) {
				n.delta[proc][n.l-k][j] = 0
				for i := 0; i < n.Sizes[n.l+2-k]; i++ {
					n.delta[proc][n.l-k][j] += n.weights[n.l+1-k][i][j] * n.delta[proc][n.l+1-k][i] *
						n.layers[n.l+1-k].prime(n.z[proc][n.l-k][j])
				}
				n.nablaB[proc][n.l-k][j] += n.delta[proc][n.l-k][j]

				for i := 0; i < n.Sizes[n.l-k]; i++ {
					n.nablaW[proc][n.l-k][j][i] += n.delta[proc][n.l-k][j] * n.activations[proc][n.l-k][i]
				}
				}(k)
			}
		}(k)
	}
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (n *Network) backPropAlgorithm(x, y []float64, proc int) {
	//defer TimeTrack(time.Now())
	defer wg.Done()

	// 1. Forward feed
	n.forwardFeed(x, proc)

	// 2. Computing the output error (delta L).
	n.outputError(y, proc)

	// 3. Gradients at the output layer
	n.outputGradients(proc)

	// 4. Backpropagating the error
	n.backPropError(proc)
}

// updateWeights updates the weight matrix following a mini batch
func (n *Network) updateWeightsSerial() {
	defer wg.Done()
	//defer TimeTrack(time.Now())
	for k := 0; k < len(n.Sizes)-1; k++ {
		go func(k int) {
			for j := 0; j < n.Sizes[k+1]; j++ {
				for i := 0; i < n.Sizes[k]; i++ {
					for idx := 1; idx < n.nCores; idx++ {
						n.nablaW[0][k][j][i] += n.nablaW[idx][k][j][i]
						n.nablaW[idx][k][j][i] = 0
					}
					n.weights[k][j][i] = (1 - n.hp.eta*(n.hp.lambda/n.data.n))*n.weights[k][j][i] -
						n.hp.eta/n.data.miniBatchSize*n.nablaW[0][k][j][i]

					// Resetting gradients to zero
					n.nablaW[0][k][j][i] = 0
				}
			}
		}(k)
	}
}

// updateWeights updates the weight matrix following a mini batch.
// The
func (n *Network) updateWeights() {
	defer wg.Done()
	//defer TimeTrack(time.Now())

	for k := 0; k < len(n.Sizes)-1; k++ {
		for _, interval := range n.layers[k+1].intervals {
			go func(intervalStart, intervalStop, k int) {
				for j := intervalStart; j < intervalStop; j++ {
					for i := 0; i < n.Sizes[k]; i++ {
						for idx := 1; idx < n.nCores; idx++ {
							n.nablaW[0][k][j][i] += n.nablaW[idx][k][j][i]
							n.nablaW[idx][k][j][i] = 0	// Resetting gradients to zero
						}

						n.weights[k][j][i] = (1-n.hp.eta*(n.hp.lambda/n.data.n))*n.weights[k][j][i] -
							n.hp.eta/n.data.miniBatchSize*n.nablaW[0][k][j][i]

						// Resetting gradients to zero
						n.nablaW[0][k][j][i] = 0
					}
				}
			}(interval[0], interval[1], k)
		}
	}
}


// updateBiases updates the bias matrix following a mini batch
func (n *Network) updateBiases() {
	defer wg.Done()

	for k := 0; k < len(n.Sizes)-1; k++ {
		for j := 0; j < n.Sizes[k+1]; j++ {
			for idx := 1; idx < n.nCores; idx++ {
				n.nablaB[0][k][j] += n.nablaB[idx][k][j]
				n.nablaB[idx][k][j] = 0
			}
			n.biases[k][j] = n.biases[k][j] - n.hp.eta/n.data.miniBatchSize*n.nablaB[0][k][j]

			// Resetting gradients to zero
			n.nablaB[0][k][j] = 0
		}
	}
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (n *Network) updateMiniBatches() {

	defer TimeTrack(time.Now())

	for i := range n.data.miniBatches {
		for idx, dataSet := range n.data.miniBatches[i] {
			wg.Add(1)
			go n.backPropAlgorithm(dataSet[0], dataSet[1], int(math.Mod(float64(idx), float64(n.nCores))))
		}

		wg.Wait()

		wg.Add(2)
		go n.updateWeightsSerial()
		go n.updateBiases()
		wg.Wait()
	}
}

// trainNetwork trains the network with the parameters given as arguments
func (n *Network) TrainNetwork(epochs int, miniBatchSize int, eta, lambda float64, shuffle, validate bool, nCores int) {
	defer TimeTrack(time.Now())

	runtime.GOMAXPROCS(nCores) // Number of cores used

	if len(n.trainingInput) == 0 || len(n.trainingOutput) == 0 {
		log.Fatal("Insufficient training data submitted")
	}

	if validate {
		if len(n.validationInput) == 0 || len(n.validationOutput) == 0 {
			log.Fatal("Insufficient validation data submitted")
		}
	}

	n.initDataContainers(nCores)
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
