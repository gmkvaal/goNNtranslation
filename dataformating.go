package network

import (
	"math/rand"
	"time"
	"github.com/gmkvaal/goNNtranslation/train/simpletest"
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

func (data *data) loadData() {
	//d := &MNIST.Data{}
	//d.FormatMNISTData()
	d := simpletest.Data{}
	d.InitTestSlices()
	data.trainingInput = d.TrainingInput
	data.trainingOutput = d.TrainingOutput
	data.validationInput = d.ValidationInput
	data.validationOutput = d.ValidationOutput
}

// initSizes initiates the fields containing the size and length of the training set and mini batch
func (data *data) initSizes(trainingSetLength int, miniBatchSize int) {
	data.n = float64(trainingSetLength)
	data.miniBatchSize = float64(miniBatchSize)
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





