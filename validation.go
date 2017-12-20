package network

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

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


func (nf NetworkFormat) validate(inputData, outputData []*mat64.Dense, dataCap int) {
	var yes, no int

	for i := range outputData[:dataCap] {
		if checkIfEqual(nf.forwardFeed(inputData[:dataCap][i]).RawMatrix().Data,
			outputData[:dataCap][i].RawMatrix().Data) == 1 {
			yes += 1
		} else {
			no += 1
		}
	}

	fmt.Println("Hitrate:", 100*float64(yes)/float64(yes+no), "%")
}

