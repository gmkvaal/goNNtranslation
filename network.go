package network

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)



// NetworkFormat contains the
// fields Sizes, biases, and weights
type NetworkFormat struct {
	Sizes       []int
	l           int
	hp          HyperParameters
	data
	NetworkMethods
	dataContainers
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

type NetworkMethods struct {
	activationFunc      func(i, j int, v float64) float64
	activationFuncPrime func(i, j int, v float64) float64
	outputErrorFunc     func(delta *mat64.Dense, a, y mat64.Matrix)
}

type HyperParameters struct {
	eta    float64
	lambda float64
}

// initNetwork initiates the weights
// and biases with random numbers
func (nf *NetworkFormat) InitNetwork() {
	nf.weights 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], randomFunc())
	nf.biases 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, randomFunc())
	nf.nablaW 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], zeroFunc())
	nf.nablaB 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.deltaNablaW	= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[:], nf.Sizes[1:], zeroFunc())
	nf.deltaNablaB 	= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.delta 		= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.z 			= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.activations 	= sliceWithGonumDense(len(nf.Sizes[:]), nf.Sizes[:], 1, zeroFunc())
	nf.sp 			= sliceWithGonumDense(len(nf.Sizes[1:]), nf.Sizes[1:], 1, zeroFunc())
	nf.l 			= len(nf.Sizes) - 1
}

func (nm *NetworkMethods) InitNetworkMethods(
	outputError func(delta *mat64.Dense, a, y mat64.Matrix),
	activationFunc func(i, j int, v float64) float64,
	activationFuncPrime func(i, j int, v float64) float64) {

	nm.outputErrorFunc = outputError
	nm.activationFunc = activationFunc
	nm.activationFuncPrime = activationFuncPrime
}

// setHyperParameters initiates the hyper parameters
func (hp *HyperParameters) InitHyperParameters(eta float64, lambda float64) {
	hp.eta = eta
	hp.lambda = lambda
}

// forwardFeed computes the z-s and activations at every neuron and returns the output layer
func (nf *NetworkFormat) forwardFeed(x mat64.Matrix) *mat64.Dense {
	nf.activations[0].Clone(x)
	for k := range nf.Sizes[1:] {
		nf.z[k].Mul(nf.weights[k].T(), nf.activations[k])
		nf.z[k].Add(nf.z[k], nf.biases[k])
		nf.activations[k+1].Apply(nf.activationFunc, nf.z[k])
	}

	return nf.activations[nf.l]
}

// outputError computes the error at the output neurons
func (nf *NetworkFormat) outputError(y mat64.Matrix) {
	nf.outputErrorFunc(nf.delta[nf.l-1], nf.activations[nf.l], y)
}

// outputGradients computes the (delta) gradients at the output layer
func (nf *NetworkFormat) outputGradients() {
	nf.deltaNablaB[nf.l-1].Clone(nf.delta[nf.l-1])
	nf.deltaNablaW[nf.l-1].Mul(nf.activations[nf.l-1], nf.delta[nf.l-1].T())
}

// backPropError backpropagates the error and computes the (delta) gradients
// at every layer
func (nf *NetworkFormat) backPropError() {
	for k := 2; k < nf.l+1; k++ {
		nf.sp[nf.l-k].Apply(nf.activationFuncPrime, nf.z[nf.l-k])
		nf.delta[nf.l-k].Mul(nf.weights[nf.l+1-k], nf.delta[nf.l+1-k])
		nf.delta[nf.l-k].MulElem(nf.delta[nf.l-k], nf.sp[nf.l-k])
		nf.deltaNablaB[nf.l-k].Clone(nf.delta[nf.l-k])
		nf.deltaNablaW[nf.l-k].Mul(nf.activations[nf.l-k], nf.delta[nf.l-k].T())
	}
}

// backProp performs one iteration of the backpropagation algorithm
// for input x and training output y (one batch in a mini batch)
func (nf *NetworkFormat) BackPropAlgorithm(x, y *mat64.Dense) {

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
func (nf *NetworkFormat) updateGradients() {
	for k := range nf.Sizes[1:] {
		nf.nablaW[k].Add(nf.nablaW[k], nf.deltaNablaW[k])
		nf.nablaB[k].Add(nf.nablaB[k], nf.deltaNablaB[k])
	}
}

// updateWeightsAtLayer updates the weights at a given layer of the network
func (nf *NetworkFormat) updateWeightAtLayer(k int) {
	nf.weights[k].Scale(1-nf.hp.eta*(nf.hp.lambda/nf.data.n), nf.weights[k])
	nf.nablaW[k].Scale(nf.hp.eta/nf.data.miniBatchSize, nf.nablaW[k])
	nf.weights[k].Sub(nf.weights[k], nf.nablaW[k])
}

// updateWeightsAtLayer updates the biases at a given layer of the network
func (nf *NetworkFormat) updateBiasesAtLayer(k int) {
	nf.nablaB[k].Scale(nf.hp.eta/nf.data.miniBatchSize, nf.nablaB[k])
	nf.biases[k].Sub(nf.biases[k], nf.nablaB[k])
}

// clearGradientsAtLayer sets the weight and bias gradients to zero
func (nf *NetworkFormat) clearGradientsAtLayer(k int) {
	nf.nablaW[k].Scale(0, nf.nablaW[k])
	nf.nablaB[k].Scale(0, nf.nablaB[k])
}

// updateWeightsAndBiases updates the weights and biases
// at every layer of the network
func (nf *NetworkFormat) updateWeightsAndBiases() {
	for k := range nf.Sizes[1:] {
		nf.updateWeightAtLayer(k)
		nf.updateBiasesAtLayer(k)
		nf.clearGradientsAtLayer(k)
	}
}

// updateMiniBatches runs the stochastic gradient descent
// algorithm for a set of mini batches (e.g one epoch)
func (nf *NetworkFormat) updateMiniBatches() {
	for i := range nf.data.miniBatches {
		for _, dataSet := range nf.data.miniBatches[i] {
			nf.BackPropAlgorithm(dataSet[0], dataSet[1])
			nf.updateGradients()
		}

		nf.updateWeightsAndBiases()
	}
}

// trainNetwork trains the network with the parameters given as arguments
func (nf *NetworkFormat) TrainNetwork(dataCap int, epochs int, miniBatchSize int, eta, lambda float64, shuffle bool) {
	nf.InitNetwork()
	nf.data.LoadData()
	nf.hp.InitHyperParameters(eta, lambda)

	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch", i, ":")

		nf.data.miniBatchGenerator(0, dataCap, miniBatchSize, shuffle)
		nf.updateMiniBatches()
		nf.validate(nf.data.validationInput, nf.data.validationOutput, 1000)

		//fmt.Println("Avg cost:", nf.totalCost(nf.data.validationInput[:dataCap], nf.data.validationInput[:dataCap]))
		fmt.Println("")
	}
}
