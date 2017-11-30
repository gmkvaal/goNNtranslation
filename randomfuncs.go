package main

import (
	"math/rand"
	"math"
)

// randomFunc returns a func that
// generates a random number
func randomFunc() func(int) float64 {
	return func(size int) float64 {
		return float64(rand.NormFloat64()) / math.Sqrt(float64(size))
	}
}

// randomFunc returns a func that returns a zero
func zeroFunc() func(int) float64 {
	return func(size int) float64 {
		return 0.0 / float64(size)
	}
}

