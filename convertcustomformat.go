package main

import (
	"github.com/petar/GoMNIST"
	"log"
)

type Data struct {
	trainingInput [][]float64
	trainingOutput [][]float64
	validationInput [][]float64
	validationOutput [][]float64
}

// customSliceToFloat64Slice converts the entries of the loaded
// MNIST data from a custom type to float64
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

// initTrainingData initiates trainingInput and trainingOutput
// with data from MNIST. trainingInput is a slice containing slices (60000)
// with data representing the pixel input (28*28). trainingOutput is a slice
// containing equally many slices of length 10, with the value 1 at the
// index corresponding to the input number (0-9).
func (data *Data)initTrainingData (train *GoMNIST.Set) {
	for i := 0; i < train.Count(); i++ {
		inputSlice, outputNumber := train.Get(i)
		data.trainingInput = append(data.trainingInput, customSliceToFloat64Slice(inputSlice))
		data.trainingOutput = append(data.trainingOutput, labelToArray(int(outputNumber)))
	}
}

// initValidationData initiates trainingInput and trainingOutput
// with data from MNIST. validationInput is a slice containing slices (60000)
// with data representing the pixel input (28*28). validationOutput is a slice
// containing equally many slices of length 10, with the value 1 at the
// index corresponding to the input number (0-9).
func (data *Data)initValidationData (test *GoMNIST.Set) {
	for i := 0; i < test.Count(); i++ {
		inputSlice, outputNumber := test.Get(i)
		data.validationInput = append(data.validationInput, customSliceToFloat64Slice(inputSlice))
		data.validationOutput = append(data.validationOutput, labelToArray(int(outputNumber)))
	}
}

// formatData loads the MNIST data and initiates the Data struct.
func (data *Data)formatData() {
	train, test, err := GoMNIST.Load("/home/guttorm/xal/go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}
	
	data.initTrainingData(train)
	data.initValidationData(test)
}