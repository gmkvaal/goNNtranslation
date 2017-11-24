package main

import (
	"fmt"
	"math/rand"
	//"github.com/petar/GoMNIST"
	//"log"
	"errors"
	"github.com/petar/GoMNIST"
	"log"

	"math"
)



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

// networkFormat contains the
// fields sizes, biases, and weights
type networkFormat struct {
	sizes []int
	biases [][]float64
	weights [][][]float64
}


// randomFunc returns a func that
// generates a random number
func randomFunc () func() float64 {
	return func() float64 {
		return float64(rand.Float64())
	}
}

// randomFunc returns a func that
// returns a zero
func zeroFunc () func() float64 {
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


// squareMatrix creates a square matrix with entries from input
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

// cubicMatrix creates a square matrix with entries from input
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

// sigmoid returns the sigmoid function
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// sigmoidPrime returns the differentiated sigmoid function
func sigmoidPrime(z float64) float64 {
	return sigmoid(z)*(1 - sigmoid(z))
}

// delta returns the error at a given neuron
func neuronError(z float64, a float64, y float64) float64 {
	return (a-y) * sigmoidPrime(z)
}

// backProp performs one iteration of the backpropagation
// algorithm for input x and training output y
func (nf networkFormat) backProp(x []float64, y [10]float64) {

	//nablaW := nf.cubicMatrix(0.0)
	//nablaB := nf.squareMatrix(0.0)

	activation := x
	activations := [][]float64{x}

	// Iterating through the layers
	for i := 0; i < len(nf.biases); i++ {
		//b :=e nf.biases[i]
		//w := nf.weights[i]

		//weightVec := mat64.NewVector(len(nf.weights[i]), nf.weights[i][1])
		//biasVec := mat64.NewVector(len(nf.biases[i]), nf.biases[i])
		//fmt.Println(weightVec.Dims())
		//fmt.Println(activationVec.Dims())
		//fmt.Println(biasVec.Dims())

		var zs []float64
		var z float64
		var sigs []float64
		var sig float64

		// Computing the activations
		// (sigmoids) for all nodes
		for idx := range nf.weights[i] {
			z = dot(activation, nf.weights[i][idx]) + nf.biases[i][idx]
			zs = append(zs, z)
			sig = sigmoid(z)
			sigs = append(sigs, sig)

		}
		activation = zs
		activations = append(activations, activation)

		var delta []float64

		for idx := range activations[len(activations)] {
			delta = append(delta, neuronError(zs[len(zs)], activations[len(activations)][idx], y[idx]))
		}

		//fmt.Println(activationVec.Dims())
		//fmt.Println(biasVec.Dims())



	}
}


func customSliceToFloat64Slice(s GoMNIST.RawImage) []float64 {

	var normalSlice []float64
	for idx:= range s {
		normalSlice = append(normalSlice, float64(s[idx]))
	}

	return normalSlice

}


func main () {

	train, _, err := GoMNIST.Load("./data")
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

	nf := &networkFormat{sizes: []int{784, 30, 10}}
	nf.setWeightsAndBiases()

	fmt.Println(x)

	nf.backProp(x, y)



	/*
	sizes := []int{786, 30, 10}
	w := make([][][]float64, len(sizes[1:]))

	for k := range w {
		w[k] = make([][]float64, sizes[k+1])

		for j := 0; j < sizes[k+1]; j++ {
			w[k][j] = make([]float64, sizes[k])

			for i := 0; i < sizes[k]; i++ {
				w[k][j][i] = rand.Float64()
			}
		}
	}


	fmt.Println(w[0])
	*/
	//nf := networkFormat{}
	//nf.initializeNetwork([784, 30, 10])

	//wb := weightsAndBiases{}
	//wb.largeWeightInitializer()
	//fmt.Println(wb.weights)

	/*
	for i := 0; i < len(nf.biases); i++ {
		//b := nf.biases[i]
		//w := nf.weights[i]

		biasVec := mat64.NewVector(len(nf.biases[i]), nf.biases[i])
		//weightVec := mat64.NewVector(len(nf.weights[i]), nf.weights[i][1])

		//fmt.Println(weightVec.Dims())
		//fmt.Println(activationVec.Dims())
		//fmt.Println(biasVec.Dims())

		for idx := range nf.weights[i] {
			weightVec := mat64.NewVector(len(nf.weights[i][idx]), nf.weights[i][idx])
			fmt.Println(weightVec.Dims())
			fmt.Println(activationVec.Dims())
			fmt.Println(biasVec.Dims())
			fmt.Println("")
		}
	*/
	}