package main

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// TestShuffleData tests that input/output data are matched after shuffling
func TestShuffleData (t *testing.T) {
	nf := &networkFormat{sizes: []int{784, 30, 10}}
	nf.initNetwork()
	nf.data.formatData()

	// Appending the index number to the last entry to match input output
	for i := 0; i < len(nf.data.trainingInput); i++ {
		nf.data.trainingInput[i] = append(nf.data.trainingInput[i], float64(i))
		nf.data.trainingOutput[i] = append(nf.data.trainingOutput[i], float64(i))
	}

	for i := 0; i < len(nf.data.validationInput); i++ {
		nf.data.validationInput[i] = append(nf.data.validationInput[i], float64(i))
		nf.data.validationOutput[i] = append(nf.data.validationOutput[i], float64(i))
	}

	nf.data.shuffleAllData()

	// Check if input/output are matched after shuffling
	for i := 0; i < len(nf.data.trainingInput); i++ {
		assert.Equal(t, nf.data.trainingInput[i][784], nf.data.trainingOutput[i][10])
	}

	for i := 0; i < len(nf.data.validationInput); i++ {
		assert.Equal(t, nf.data.validationInput[i][784], nf.data.validationOutput[i][10])
	}
}
