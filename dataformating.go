package network

import (
	"github.com/gmkvaal/goNNtranslation/train/MNIST"
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"time"
	"fmt"
)

type data struct {
	trainingInput    []*mat64.Dense
	trainingOutput   []*mat64.Dense
	validationInput  []*mat64.Dense
	validationOutput []*mat64.Dense
	miniBatches      [][][]*mat64.Dense
	n                float64
	miniBatchSize    float64
}

func (data *data) LoadData() {
	d := &MNIST.Data{}
	d.FormatMNISTData()
	//d := simpletest.Data{}
	//d.InitTestSlices()
	for idx := range d.TrainingInput {
		data.trainingInput = append(data.trainingInput,
			mat64.NewDense(len(d.TrainingInput[idx]), 1, d.TrainingInput[idx]))
		data.trainingOutput = append(data.trainingOutput,
			mat64.NewDense(len(d.TrainingOutput[idx]), 1, d.TrainingOutput[idx]))
	}

	for idx := range d.ValidationOutput {
		data.validationInput = append(data.trainingInput,
			mat64.NewDense(len(d.ValidationInput[idx]), 1, d.ValidationInput[idx]))
		data.validationOutput = append(data.trainingOutput,
			mat64.NewDense(len(d.ValidationOutput[idx]), 1, d.ValidationOutput[idx]))
	}

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
	numberOfMiniBatches := int(trainingSetLength / miniBatchSize)
	data.miniBatches = make([][][]*mat64.Dense, numberOfMiniBatches, numberOfMiniBatches)
	data.initSizes(trainingSetLength, miniBatchSize)

	for i := 0; i < numberOfMiniBatches; i++ {
		data.miniBatches[i] = make([][]*mat64.Dense, miniBatchSize, miniBatchSize)
		for j := 0; j < miniBatchSize; j++ {
			data.miniBatches[i][j] = []*mat64.Dense{data.trainingInput[i*miniBatchSize+j],
				data.trainingOutput[i*miniBatchSize+j]}
		}
	}
}

