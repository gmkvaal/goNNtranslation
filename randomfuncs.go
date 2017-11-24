package main

import 	"math/rand"

// randomFunc returns a func that
// generates a random number
func randomFunc() func() float64 {
	return func() float64 {
		return float64(rand.NormFloat64())
	}
}

// randomFunc returns a func that returns a zero
func zeroFunc() func() float64 {
	return func() float64 {
		return 0.0
	}
}

