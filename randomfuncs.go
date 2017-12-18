package network

import (
	"math/rand"
	"time"
)

// randomFunc returns a func that
// generates a random number
func randomFunc() func(int) float64 {
	rand.Seed(time.Now().UnixNano())
	return func(size int) float64 {
		return float64(rand.Float64()) // math.Sqrt(float64(size))
	}
}

// randomFunc returns a func that returns a zero
func zeroFunc() func(int) float64 {
	return func(size int) float64 {
		return 0.0 / float64(size)
	}
}

func oneFunc() func(int) float64 {
	return func(size int) float64 {
		return 1
	}
}