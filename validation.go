package main

import "fmt"

// argMax returns the index corresponding
// to the largest entry in slice s
func argMax (s []float64) int {

	largestNumberIdx := 0
	largestNumber := 0.0
	for idx, val := range s {
		if idx == 0 {
			largestNumber = val
		}
		if val > largestNumber {
			largestNumberIdx = idx
			largestNumber = val
		} else {
			largestNumberIdx = largestNumberIdx
		}
	}

	return largestNumberIdx
}

// validate checks if the output from the network matches
// the the validation output. Returns 1 if true, 0 if false
func checkIfEqual(yNetwork []float64, y []float64) int {

	if argMax(yNetwork) == argMax(y) {
		return 1
	} else {
		return 0
	}
}

func (nf networkFormat) validate(dataCap int, inputData [][]float64, outputData [][]float64) {
	var yes, no int
	for i := range outputData[:dataCap] {
		if checkIfEqual(nf.forwardFeedValidation(inputData[:dataCap][i]), outputData[:dataCap][i]) == 1 {
			yes += 1
		} else {
			no += 1
		}
	}

	fmt.Println(yes, no, float64(yes)/float64(yes+no))
}

func (nf networkFormat) forwardFeedValidation(x []float64) []float64 {
	// Clearing / preparing the slices
	l := len(nf.sizes) - 1 // last entry "layer-vise"

	z := nf.squareMatrix(zeroFunc())
	activations := nf.squareMatrixFull(zeroFunc())
	activations[0] = x

	// Forward feed
	for k := 0; k < l; k++ {
		for j := 0; j < nf.sizes[k+1]; j++ {
			for i := 0; i < nf.sizes[k]; i++ {
				z[k][j] += activations[k][i] * nf.weights[k][j][i]
			}
			activations[k+1][j] = sigmoid(z[k][j] + nf.biases[k][j])
		}
	}

	return activations[2]
}