package main

import (
	"errors"
	"fmt"
	"github.com/petar/GoMNIST"
	"log"
	"math/rand"

	"math"
)

// NOTE: Maybe not so smart with all these global variables?

// Variables to hold the activation(s) for one node,
// nodes on one layer, and all layers, respectively
var zNode float64
var zLayer []float64
var zAllLayers [][]float64

// Variables to hold the sigmoid(s) for one node,
// nodes on a new layer, input layer, and every
// layer, respectively.
var activationNode float64
var activationNewLayer []float64
var activationLayer []float64
var activationAllLayers [][]float64

var deltaOutputLayer []float64

var nablaW [][][]float64
var nablaB [][]float64

// networkFormat contains the
// fields sizes, biases, and weights
type networkFormat struct {
	sizes   []int
	biases  [][]float64
	weights [][][]float64
}

// randomFunc returns a func that
// generates a random number
func randomFunc() func() float64 {
	return func() float64 {
		return float64(rand.NormFloat64())
	}
}

// randomFunc returns a func that returns a zero
func zeroFunc() func() float64 {
	return func() float64 {
		return 0.0
	}
}

// setWeightsAndBiases initiates the weights
// and biases with random numbers
func (nf *networkFormat) setWeightsAndBiases() {
	nf.weights = nf.cubicMatrix(randomFunc())
	nf.biases = nf.squareMatrix(randomFunc())
}

// squareMatrix creates a square matrix with entries from input function
func (nf networkFormat) squareMatrix(input func() float64) [][]float64 {
	b := make([][]float64, len(nf.sizes[1:]))

	for k := range b {
		b[k] = make([]float64, nf.sizes[k+1])

		for j := 0; j < nf.sizes[k+1]; j++ {
			b[k][j] = input()
		}
	}

	return b
}

// cubicMatrix creates a square matrix with entries from input function
func (nf networkFormat) cubicMatrix(input func() float64) [][][]float64 {
	w := make([][][]float64, len(nf.sizes[1:]))

	for k := range w {
		w[k] = make([][]float64, nf.sizes[k+1])

		for j := 0; j < nf.sizes[k+1]; j++ {
			w[k][j] = make([]float64, nf.sizes[k])

			for i := 0; i < nf.sizes[k]; i++ {
				w[k][j][i] = input()
			}
		}
	}

	return w
}

// dot performs the dot product of two slices a and b
func dot(a []float64, b []float64) float64 {
	var product float64
	for idx := range a {
		product += a[idx] + b[idx]
	}

	return product
}

func vectorMatrixProduct(matrix [][]float64, a []float64, b []float64) [][]float64{
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(b); j++ {
			matrix[j][i] = a[i]*b[j]
		}
	}

	return matrix
}

// sigmoid returns the sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// sigmoidPrime returns the differentiated sigmoid function
func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

// delta returns the error at a given neuron
func outputNeuronError(z float64, a float64, y float64) float64 {
	return (a - y) * sigmoidPrime(z)
}

// forwardFeed updates all neurons from the activations. Depends on being
// called inside the scope of backProp where activationLayers are defined (dangerous?)
func (nf networkFormat) forwardFeed () ([][]float64, [][]float64){

	// Iterating through the layers
	for k := 0; k < len(nf.biases); k++ {

		// Computing the activations for nodes on a single layer
		// NB: NEED TO SCALE IN THE INPUT SO THAT THE NEURONS
		// DOES NOT GET INSTANTLY SATURATED
		for idx := range nf.weights[k] {
			zNode = dot(activationLayer, nf.weights[k][idx]) + nf.biases[k][idx]
			zLayer = append(zLayer, zNode)
			activationNode = sigmoid(zNode)
			activationNewLayer = append(activationNewLayer, activationNode)
		}

		activationLayer = activationNewLayer
		zAllLayers = append(zAllLayers, zLayer)
		activationAllLayers = append(activationAllLayers, activationLayer)

		// Clearing the slices
		zLayer = nil
		activationNewLayer = nil
	}

	return zAllLayers, activationAllLayers
}

// outputError computes the output error (delta L) by Looping over output neurons.
func (nf networkFormat) outputError (y [10]float64) []float64 {
	for idx := range activationAllLayers[len(activationAllLayers)-1] {
		z := zAllLayers[len(zAllLayers)-1][idx]
		a := activationAllLayers[len(activationAllLayers)-1][idx]
		deltaOutputNeuron := outputNeuronError(z, a, y[idx])
		deltaOutputLayer = append(deltaOutputLayer, deltaOutputNeuron)
	}

	return deltaOutputLayer

}

// backProp performs one iteration of the backpropagation
// algorithm for input x and training output y
func (nf networkFormat) backProp(x []float64, y [10]float64) {

	// Initiating the gradient matrices
	nablaW = nf.cubicMatrix(zeroFunc())
	nablaB = nf.squareMatrix(zeroFunc())
	l := len(nf.sizes) - 1// last entry "layer-vise"
	lNab := len(nablaW) -1 // last entry
	fmt.Println("l is", l)

	// Clearing / preparing the slices
	zAllLayers = nil
	activationLayer = x
	activationAllLayers = [][]float64{x}

	// Updating all neurons with the forwardFeed algorithmn
	zAllLayers, activationAllLayers = nf.forwardFeed()

	// Computing the output error (delta L).
	delta := nf.outputError(y)
	nablaB[lNab] = delta
	nablaW[lNab] = vectorMatrixProduct(nablaW[lNab], activationAllLayers[l-1], delta)








}

func customSliceToFloat64Slice(s GoMNIST.RawImage) []float64 {

	// Divinding on 255.0 to scale the
	// input into the range 0 - 1.
	var normalSlice []float64
	for idx := range s {
		normalSlice = append(normalSlice, float64(s[idx])/255.0)
	}

	return normalSlice
}

// labelToArray converts the base integers into an
// array with '1' at the respective entry, rest 0
func labelToArray(label int) ([10]float64, error) {

	tennerArray := [10]float64{}

	if label > 9 {
		return tennerArray, errors.New("wrong format - can only convert numbers < 9")
	}

	tennerArray[label] = 1

	return tennerArray, nil
}

func main() {


	// Load files of type GoMNIST which is actually []byte, where the byte value
	train, _, err := GoMNIST.Load("/home/guttorm/xal/go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}

	sweeper := train.Sweep()
	image, label, _ := sweeper.Next()

	labelArr, err := labelToArray(int(label))
	if err != nil {
		log.Fatal(err)
	}

	y := labelArr
	x := customSliceToFloat64Slice(image)

	fmt.Println("")

	nf := &networkFormat{sizes: []int{784, 30, 10}}
	nf.setWeightsAndBiases()
	nf.backProp(x, y)

}
