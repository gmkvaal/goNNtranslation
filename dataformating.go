package main

import (
	"github.com/petar/GoMNIST"
	"log"
	"math/rand"
	"time"
)

type data struct {
	trainingInput [][]float64
	trainingOutput [][]float64
	validationInput [][]float64
	validationOutput [][]float64
	miniBatches [][][][]float64
	n float64
	miniBatchSize float64
}

// initTrainingData initiates trainingInput and trainingOutput
// with data from MNIST. trainingInput is a slice containing slices (60000)
// with data representing the pixel input (28*28). trainingOutput is a slice
// containing equally many slices of length 10, with the value 1 at the
// index corresponding to the input number (0-9).
func (data *data) initTrainingData(train *GoMNIST.Set) {
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
func (data *data) initValidationData(test *GoMNIST.Set) {
	for i := 0; i < test.Count(); i++ {
		inputSlice, outputNumber := test.Get(i)
		data.validationInput = append(data.validationInput, customSliceToFloat64Slice(inputSlice))
		data.validationOutput = append(data.validationOutput, labelToArray(int(outputNumber)))
	}
}

// initSizes initiates the fields containing the size and length of the training set and mini batch
func (data *data) initSizes(trainingSetLength int, miniBatchSize int) {
	data.n = float64(trainingSetLength)
	data.miniBatchSize = float64(miniBatchSize)
}

// formatData loads the MNIST data and initiates the Data struct.
func (data *data) formatData() {
	train, test, err := GoMNIST.Load("/home/guttorm/xal/go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}
	
	data.initTrainingData(train)
	data.initValidationData(test)
}

// shuffleTrainingData shuffles the training data
func (data *data) shuffleTrainingData() {
	for i := len(data.trainingInput) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data.trainingInput[i], data.trainingInput[j] = data.trainingInput[j], data.trainingInput[i]
		data.trainingOutput[i], data.trainingOutput[j] = data.trainingOutput[j], data.trainingOutput[i]
	}
}

// shuffleValidationData shuffles the validation data
func (data *data) shuffleValidationData() {
	for i := len(data.validationInput) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		data.validationInput[i], data.validationInput[j] = data.validationInput[j], data.validationInput[i]
		data.validationOutput[i], data.validationOutput[j] = data.validationOutput[j], data.validationOutput[i]
	}
}

// shuffleAllData sets a unique seed to initiate unique shuffles
// and calls shuffleTrainingData and suffleValidationData
func (data *data) shuffleAllData() {
	rand.Seed(time.Now().UnixNano())
	data.shuffleTrainingData()
	data.shuffleValidationData()
}

// miniBatchGenerator generates a new set of miniBatches from the training data.
// miniBatches contain (numberOfMiniBatches) number of mini batches, each of which contains (miniBatchSize) number
// of len 2 slices containing the trainingInput and trainingOutput at the respective entries.
func (data *data) miniBatchGenerator(dataStart, dataCap, miniBatchSize int, shuffle bool) {

	dataStart = 0 // not yet implemented

	if shuffle {
		data.shuffleAllData()
	}


	trainingSetLength := len(data.trainingInput[dataStart:dataCap])
	numberOfMiniBatches := int(trainingSetLength/miniBatchSize)
	data.miniBatches = make([][][][]float64, numberOfMiniBatches, numberOfMiniBatches)
	data.initSizes(trainingSetLength, miniBatchSize)

	for i := 0; i < numberOfMiniBatches; i++ {
		data.miniBatches[i] = make([][][]float64, miniBatchSize, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			data.miniBatches[i][j] = [][]float64{data.trainingInput[i*miniBatchSize + j],
				data.trainingOutput[i*miniBatchSize + j]}
		}
	}
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
