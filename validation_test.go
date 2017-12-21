package network

import (
	"github.com/stretchr/testify/assert"
	"testing"
	"github.com/gonum/matrix/mat64"
)

func TestArgMax(t *testing.T) {
	slice := []float64{1.0, 3.0, 2.0}
	assert.Equal(t, argMax(slice), 1)
}

func TestCheckIfEqual(t *testing.T) {
	slice1 := []float64{1.0, 2.0, 3.0}
	slice2 := []float64{1.0, 2.0, 3.0}
	assert.Equal(t, checkIfEqual(slice1, slice2), 1)

	slice1 = []float64{1.0, 5.0, 3.0}
	slice2 = []float64{1.0, 2.0, 3.0}
	assert.Equal(t, checkIfEqual(slice1, slice2), 0)
}

func TestValidateArgMaxSlice(t *testing.T) {
	n := Network{}

	dense1 := mat64.NewDense(3,1, []float64{1.0, 2.0, 3.0})
	dense2 := mat64.NewDense(3,1, []float64{2.0, 2.0, 3.0})
	dense3 := mat64.NewDense(3,1, []float64{3.0, 2.0, 3.0})

	inputData := []*mat64.Dense{dense1, dense2, dense3}
	outputData := []*mat64.Dense{dense1, dense2, dense3}

	assert.Equal(t, n.ValidateArgMaxSlice(inputData, outputData), true)

	inputData = []*mat64.Dense{dense1, dense3, dense3}
	outputData = []*mat64.Dense{dense1, dense2, dense3}

	assert.Equal(t, n.ValidateArgMaxSlice(inputData, outputData), false)

}

