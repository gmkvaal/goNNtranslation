package main

import (
	"github.com/petar/GoMNIST"
	"log"
	"fmt"
)

type trainingData struct {
	trainingInput [][]float64
	trainingOutput [][]float64
}

type validationData struct {
	validationInput [][]float64
	validationOutput [][]float64
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
func labelToArray(label int) ([]float64) {

	tennerArray := make([]float64, 10.0, 10.0)

	if label > 9 {
		return tennerArray
	}

	tennerArray[label] = 1

	return tennerArray
}

func (td *trainingData)initTrainingData (train *GoMNIST.Set) {
	for i := 0; i < train.Count(); i++ {
		inputSlice, outputNumber := train.Get(i)
		td.trainingInput = append(td.trainingInput, customSliceToFloat64Slice(inputSlice))
		td.trainingOutput = append(td.trainingOutput, labelToArray(int(outputNumber)))
	}
}

func (td *trainingData)formatData() {
	train, test, err := GoMNIST.Load("/home/guttorm/xal/go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(train.Count())
	fmt.Println(test.Count())

	td.initTrainingData(train)
}