package network

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
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
	outputErrorFunc     func(delta *mat64.Dense, a, y mat64.Matrix)
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
func (nf *Network) initDataContainers() {
	nf.setSizes()
	nf.weights 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], randomFunc())
	nf.biases 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, randomFunc())
	nf.nablaW 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], zeroFunc())
	nf.nablaB 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.deltaNablaW	= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], zeroFunc())
	nf.deltaNablaB 	= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.delta 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.z 			= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.activations 	= sliceWithGonumDense(len(nf.Sizes[:]),  nf.Sizes[:], 1, zeroFunc())
	nf.sp 			= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.l 			= len(nf.Sizes) - 1
}

func (nm *NetworkMethods) InitNetworkMethods(outputError func(delta *mat64.Dense, a, y mat64.Matrix)) {
	nm.outputErrorFunc = outputError

}

// setHyperParameters initiates the hyper parameters
func (hp *HyperParameters) InitHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed computes the z-s and activations at every neuron and returns the output layer
func (nf *Network) forwardFeed(x mat64.Matrix) *mat64.Dense {
	nf.activations[0].Clone(x)
	for k := range nf.Sizes[1:] {

		nf.z[k].Mul(nf.weights[k].T(), nf.activations[k])
		nf.z[k].Add(nf.z[k], nf.biases[k])
		nf.activations[k+1].Apply(nf.layers[k].activationFunction.function, nf.z[k])
	}

	return nf.activations[nf.l]
}

// outputError computes the error at the output neurons
func (nf *Network) outputError(y mat64.Matrix) {
	nf.outputErrorFunc(nf.delta[nf.l-1], nf.activations[nf.l], y)
}

// outputGradients computes the (delta) gradients at the output layer
func (nf *Network) outputGradients() {
	nf.deltaNablaB[nf.l-1].Clone(nf.delta[nf.l-1])
	nf.deltaNablaW[nf.l-1].Mul(nf.activations[nf.l-1], nf.delta[nf.l-1].T())
}

// backPropError backpropagates the error and computes the (delta) gradients
// at every layer
func (nf *Network) backPropError() {
	for k := 2; k < nf.l+1; k++ {
		nf.sp[nf.l-k].Apply(nf.layers[k].activationFunction.prime, nf.z[nf.l-k])
		nf.delta[nf.l-k].Mul(nf.weights[nf.l+1-k], nf.delta[nf.l+1-k])
		nf.delta[nf.l-k].MulElem(nf.delta[nf.l-k], nf.sp[nf.l-k])
		nf.deltaNablaB[nf.l-k].Clone(nf.delta[nf.l-k])
		nf.deltaNablaW[nf.l-k].Mul(nf.activations[nf.l-k], nf.delta[nf.l-k].T())
	}
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (nf *Network) BackPropAlgorithm(x, y *mat64.Dense) {

	// 1. Forward feed
	nf.forwardFeed(x)

	// 2. Computing the output error (delta L).
	nf.outputError(y)

	// 3. Gradients at the output layer
	nf.outputGradients()

	// 4. Backpropagating the error
	nf.backPropError()
}

// updateGradients adds the delta gradient matrices to the gradient matrices
func (nf *Network) updateGradients() {
	for k := range nf.Sizes[1:] {
		nf.nablaW[k].Add(nf.nablaW[k], nf.deltaNablaW[k])
		nf.nablaB[k].Add(nf.nablaB[k], nf.deltaNablaB[k])
	}
}

// updateWeightsAtLayer updates the weights at a given layer of the network
func (nf *Network) updateWeightAtLayer(k int) {
	nf.weights[k].Scale(1-nf.hp.eta*(nf.hp.lambda/nf.data.n), nf.weights[k])
	nf.nablaW[k].Scale(nf.hp.eta/nf.data.miniBatchSize, nf.nablaW[k])
	nf.weights[k].Sub(nf.weights[k], nf.nablaW[k])
}

// updateWeightsAtLayer updates the biases at a given layer of the network
func (nf *Network) updateBiasesAtLayer(k int) {
	nf.nablaB[k].Scale(nf.hp.eta/nf.data.miniBatchSize, nf.nablaB[k])
	nf.biases[k].Sub(nf.biases[k], nf.nablaB[k])
}

// clearGradientsAtLayer sets the weight and bias gradients to zero
func (nf *Network) clearGradientsAtLayer(k int) {
	nf.nablaW[k].Scale(0, nf.nablaW[k])
	nf.nablaB[k].Scale(0, nf.nablaB[k])
}

// updateWeightsAndBiases updates the weights and biases
// at every layer of the network
func (nf *Network) updateWeightsAndBiases() {
	for k := range nf.Sizes[1:] {
		nf.updateWeightAtLayer(k)
		nf.updateBiasesAtLayer(k)
		nf.clearGradientsAtLayer(k)
	}
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (nf *Network) updateMiniBatches() {
	for i := range nf.data.miniBatches {
		for _, dataSet := range nf.data.miniBatches[i] {
			nf.BackPropAlgorithm(dataSet[0], dataSet[1])
			nf.updateGradients()
		}

		nf.updateWeightsAndBiases()
	}
}

// trainNetwork trains the network with the parameters given as arguments
func (nf *Network) TrainNetwork(epochs int, miniBatchSize int, eta, lambda float64, shuffle bool) {

	nf.initDataContainers()
	nf.hp.InitHyperParameters(eta, lambda)

	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch", i, ":")

		nf.data.miniBatchGenerator(miniBatchSize, shuffle)
		nf.updateMiniBatches()
		nf.validate(nf.data.validationInput, nf.data.validationOutput)

		//fmt.Println("Avg cost:", nf.totalCost(nf.data.validationInput[:dataCap], nf.data.validationInput[:dataCap]))
		fmt.Println("")
	}
}
