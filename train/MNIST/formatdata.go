package MNIST

import (
	"github.com/petar/GoMNIST"
	"log"
)


type Data struct {
	TrainingInput    [][]float64
	TrainingOutput   [][]float64
	ValidationInput  [][]float64
	ValidationOutput [][]float64
}
// initTrainingData initiates trainingInput and trainingOutput
// with data from MNIST. trainingInput is a slice containing slices (60000)
// with data representing the pixel input (28*28). trainingOutput is a slice
// containing equally many slices of length 10, with the value 1 at the
// index corresponding to the input number (0-9).
func (data *Data) initTrainingData(train *GoMNIST.Set) {
	for i := 0; i < train.Count(); i++ {
		inputSlice, outputNumber := train.Get(i)
		data.TrainingInput = append(data.TrainingInput, customSliceToFloat64Slice(inputSlice))
		data.TrainingOutput = append(data.TrainingOutput, LabelToArray(int(outputNumber)))
	}
}

// initValidationData initiates trainingInput and trainingOutput
// with data from MNIST. validationInput is a slice containing slices (60000)
// with data representing the pixel input (28*28). validationOutput is a slice
// containing equally many slices of length 10, with the value 1 at the
// index corresponding to the input number (0-9).
func (data *Data) initValidationData(test *GoMNIST.Set) {
	for i := 0; i < test.Count(); i++ {
		inputSlice, outputNumber := test.Get(i)
		data.ValidationInput = append(data.ValidationInput, customSliceToFloat64Slice(inputSlice))
		data.ValidationOutput = append(data.ValidationOutput, LabelToArray(int(outputNumber)))
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

// LabelToArray converts the base integers into an
// array with '1' at the respective entry, rest 0
func LabelToArray(label int) ([]float64) {

	tennerArray := make([]float64, 10.0, 10.0)

	if label > 9 {
		return tennerArray
	}

	tennerArray[label] = 1

	return tennerArray
}

// formatData loads the MNIST data and initiates the Data struct.
func (data *Data) FormatMNISTData() {
	train, test, err := GoMNIST.Load("/home/guttorm/xal/go/src/github.com/petar/GoMNIST/data")
	if err != nil {
		log.Fatal(err)
	}

	data.initTrainingData(train)
	data.initValidationData(test)
}

