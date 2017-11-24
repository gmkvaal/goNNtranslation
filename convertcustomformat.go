package main

import (
	"github.com/petar/GoMNIST"
	"errors"
)

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

