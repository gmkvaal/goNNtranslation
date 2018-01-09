package network

import (
	"testing"
	"fmt"

	plr "github.com/gmkvaal/pythonlistreader"
)


// TestForwardFeed tests the forward feed algorithm
// by initiating with zero-weights and biases, hence making
// all z's zero and thus all activations 1/2 (given sigmoids)






// TestBackProp tests the back propagation algorithm
// by running through two mini batches of length two,
// and then comparing the weights and biases with ones from the
// validated solver https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src/network2
// Weights and biases are for the purpose of creating
// a deterministic result initiated as 1's.
func TestBackProp(t *testing.T) {
	n := Network{}
	n.AddLayer(784, Sigmoid, SigmoidPrime, []int{784})
	n.AddLayer(30, Sigmoid, SigmoidPrime, []int{30})
	n.AddLayer(10, Sigmoid, SigmoidPrime, []int{10})
	n.InitNetworkMethods(OutputNeuronError, ValidateArgMaxSlice)

	n.setSizes()
	n.l = len(n.Sizes) - 1
	n.nCores = 1
	n.weights = n.cubicMatrix(oneFunc())
	n.biases = n.squareMatrix(oneFunc())
	for idx := 0; idx < n.nCores; idx++ {
		n.delta = append(n.delta, n.squareMatrix(zeroFunc()))
		n.nablaW = append(n.nablaW, n.cubicMatrix(zeroFunc()))
		n.nablaB = append(n.nablaB, n.squareMatrix(zeroFunc()))
		n.z = append(n.z, n.squareMatrix(zeroFunc()))
		n.activations = append(n.activations, n.squareMatrixFull(zeroFunc()))

		n.miniBatchSize = 2
		n.n = 4
		n.hp.eta = 1
		n.hp.lambda = 5.0

		x1 := make([]float64, 784, 784)
		y1 := make([]float64, 10, 10)
		y1[0] = 1
		x1[1] = 1
		b1 := [][]float64{x1, y1}

		x2 := make([]float64, 784, 784)
		y2 := make([]float64, 10, 10)
		y2[1] = 1
		x2[2] = 1
		b2 := [][]float64{x2, y2}

		x3 := make([]float64, 784, 784)
		y3 := make([]float64, 10, 10)
		y3[2] = 1
		x3[3] = 1
		b3 := [][]float64{x3, y3}

		x4 := make([]float64, 784, 784)
		y4 := make([]float64, 10, 10)
		y4[3] = 1
		x4[4] = 1
		b4 := [][]float64{x4, y4}

		miniBatchA := [][][]float64{b1, b2}
		miniBatchB := [][][]float64{b3, b4}
		miniBatches := [][][][]float64{miniBatchA, miniBatchB}
		n.data.miniBatches = miniBatches
		n.updateMiniBatches()

		testData, err := plr.ReadFile("testdata/bias1.txt")
		if err != nil {
			t.Fatal(err)
		}

		bias1FromPy := plr.PythonFloatListParser(testData)

		testData, err = plr.ReadFile("testdata/bias0.txt")
		if err != nil {
			t.Fatal(err)
		}

		bias0FromPy := plr.PythonFloatListParser(testData)

		testData, err = plr.ReadFile("testdata/weights1.txt")
		if err != nil {
			t.Fatal(err)
		}

		weights1FromPy := plr.PythonNestedFloatListParser(testData)

		testData, err = plr.ReadFile("testdata/weights0.txt")
		if err != nil {
			t.Fatal(err)
		}

		weights0FromPy := plr.PythonNestedFloatListParser(testData)
		//fmt.Println(len(weights1FromPy))
		//fmt.Println(len(n.weights[1]))
		

		for idx := range bias1FromPy {
			if bias1FromPy[idx]-n.biases[1][idx] > 1e-7 {
				t.Error("not equal", bias1FromPy[idx], n.biases[1][idx])
			}
		}

		for idx := range bias0FromPy {
			if bias0FromPy[idx]-n.biases[0][idx] > 1e-7 {
				t.Error("not equal", bias0FromPy[idx], n.biases[0][idx])
			}
		}

		for idx1 := range weights1FromPy {
			for idx2 := range weights1FromPy[idx1] {
				if weights1FromPy[idx1][idx2]-n.weights[1][idx1][idx2] > 1e-7 {
					t.Error("not equal", weights1FromPy[idx1][idx2], n.weights[1][idx1][idx2])
				}
			}
		}

		for idx1 := range weights0FromPy {
			for idx2 := range weights0FromPy[idx1] {
				if weights0FromPy[idx1][idx2]-n.weights[0][idx1][idx2] > 1e-7 {
					t.Error("not equal", weights0FromPy[idx1][idx2], n.weights[0][idx1][idx2])
				}
			}
		}
	}
}







func vizNumber(s []float64) {
	for idx := range s {
		if s[idx] > 0 {
			s[idx] = 1
		}
	}

	counter := 0
	matrix := make([][]float64, 28, 28)
	for i := 0; i < 28; i++ {
		matrix[i] = make([]float64, 28, 28)
		for j := 0; j < 28; j++ {
			matrix[i][j] = s[counter]
			counter++
		}
	}

	for i := 0; i < 28; i++ {
		fmt.Println(matrix[i])
	}
}


