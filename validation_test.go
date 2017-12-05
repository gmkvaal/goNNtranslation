package main

import (
	"github.com/stretchr/testify/assert"
	"testing"
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

/*
func TestValidate(t *testing.T) {
	slice1 := []float64{1.0, 2.0, 3.0}
	slice2 := []float64{1.0, 2.0, 3.0}
	slice3 := []float64{1.0, 2.0, 3.0}

	bigSlice1 := [][]float64{slice1, slice2, slice3}
	bigSlice2 := [][]float64{slice1, slice2, slice3}

	assert.Equal()

}
*/