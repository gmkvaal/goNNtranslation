package main

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

var nf *networkFormat

func initNetworkForTesting() *networkFormat {
	nf = &networkFormat{sizes: []int{784, 30, 10}}
	nf.initNetwork()
	nf.formatData()
	return nf
}


func TestLabelToArray(t *testing.T) {
	n1 := []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	n2 := []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	n3 := []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	n4 := []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
	n5 := []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}
	n6 := []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
	n7 := []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}
	n8 := []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
	n9 := []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
	n10 := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}

	N := [][]float64{n1,n2,n3,n4,n5,n6,n7,n8,n9,n10}
	for label, slice := range N {
		assert.Equal(t, labelToArray(label), slice)
	}

}


// TestShuffleData tests that input/output data are matched after shuffling
func TestShuffleAllData(t *testing.T) {
	nf = initNetworkForTesting()

	// Testing that all data is read
	func (t *testing.T) {
		assert.Equal(t, len(nf.trainingInput), 60000)
		assert.Equal(t, len(nf.trainingOutput), 60000)
		assert.Equal(t, len(nf.validationInput), 10000)
		assert.Equal(t, len(nf.validationOutput), 10000)
	}(t)

	// Appending the index number to the last entry to match input output
	for i := 0; i < len(nf.trainingInput); i++ {
		nf.trainingInput[i] = append(nf.trainingInput[i], float64(i))
		nf.trainingOutput[i] = append(nf.trainingOutput[i], float64(i))
	}

	for i := 0; i < len(nf.validationInput); i++ {
		nf.validationInput[i] = append(nf.validationInput[i], float64(i))
		nf.validationOutput[i] = append(nf.validationOutput[i], float64(i))
	}

	nf.shuffleAllData()

	// Check if input/output are matched after shuffling
	for i := 0; i < len(nf.trainingInput); i++ {
		assert.Equal(t, nf.trainingInput[i][784], nf.trainingOutput[i][10])
	}

	for i := 0; i < len(nf.validationInput); i++ {
		assert.Equal(t, nf.validationInput[i][784], nf.validationOutput[i][10])
	}
}
