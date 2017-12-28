package network

import (
	"testing"

	plr "github.com/gmkvaal/pythonlistreader"
	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
	"fmt"
	"math"
)


// TestForwardFeed tests the forward feed algorithm
// by initiating with zero-weights and biases, hence making
// all z's zero and thus all activations 1/2 (given sigmoids)
func TestForwardFeed(t *testing.T) {
	n := Network{}
	n.AddLayer(2, Sigmoid, SigmoidPrime)
	n.AddLayer(3, Sigmoid, SigmoidPrime)
	n.AddLayer(1, Sigmoid, SigmoidPrime)
	n.initDataContainers()

	w1 := mat64.NewDense(2,3, nil)
	w1.Set(0, 0, -0.95766323)
	w1.Set(1, 0, -2.83527046)
	w1.Set(0, 1, -4.68051798)
	w1.Set(1, 1, -3.67367494)
	w1.Set(0, 2, -3.42136137)
	w1.Set(1, 2, -3.66026809)

	w2 := mat64.NewDense(3,1, nil)
	w2.Set(0, 0,  3.23952935)
	w2.Set(1, 0,  9.16831414)
	w2.Set(2, 0,  7.20927857)

	n.weights = []*mat64.Dense{w1, w2}

	b1 := mat64.NewDense(3, 1, nil)
	b1.Set(0, 0, 1.47931576)
	b1.Set(1, 0, 5.76116679)
	b1.Set(2, 0, 4.77665241)

	b2 := mat64.NewDense(1,1,nil)
	b2.Set(0,0,-7.41086319)

	n.biases = []*mat64.Dense{b1, b2}

	var y *mat64.Dense
	y = n.forwardFeed(mat64.NewDense(2,1, []float64{0,0}))
	assert.Equal(t, y.RawMatrix().Data[0], 0.9999900331454943)
	y = n.forwardFeed(mat64.NewDense(2,1, []float64{1,0}))
	assert.Equal(t, y.RawMatrix().Data[0], 0.9992529245275563)
	y = n.forwardFeed(mat64.NewDense(2,1, []float64{0,1}))
	assert.Equal(t, y.RawMatrix().Data[0], 0.998931751778988)
	y = n.forwardFeed(mat64.NewDense(2,1, []float64{1,1}))
	assert.Equal(t, y.RawMatrix().Data[0], 0.002937291674019087)
}




// TestBackProp tests the back propagation algorithm
// by running through two mini batches of length two,
// and then comparing the weights and biases with ones from the
// validated solver https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src/network2
// Weights and biases are for the purpose of creating
// a deterministic result initiated as 1's.
func TestBackProp(t *testing.T) {
	n := Network{}
	n.AddLayer(784, Sigmoid, SigmoidPrime)
	n.AddLayer(30, Sigmoid, SigmoidPrime)
	n.AddLayer(10, Sigmoid, SigmoidPrime)
	n.InitNetworkMethods(OutputErrorXEntropy)

	n.setSizes()
	n.weights 		= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[:], n.Sizes[1:], oneFunc())
	n.biases 		= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, oneFunc())
	n.nablaW 		= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[:], n.Sizes[1:], zeroFunc())
	n.nablaB 		= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, zeroFunc())
	n.deltaNablaW	= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[:], n.Sizes[1:], zeroFunc())
	n.deltaNablaB 	= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, zeroFunc())
	n.delta 		= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, zeroFunc())
	n.z 			= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, zeroFunc())
	n.activations 	= sliceWithGonumDense(len(n.Sizes[:]),  n.Sizes[:], 1, zeroFunc())
	n.sp 			= sliceWithGonumDense(len(n.Sizes[1:]), n.Sizes[1:], 1, zeroFunc())
	n.l 			= len(n.Sizes) - 1

	n.miniBatchSize = 2
	n.n = 4
	n.hp.eta = 1
	n.hp.lambda = 5.0

	x := make([]float64, 784, 784)
	x[1] = 1
	x1 := mat64.NewDense(784, 1, x)
	y := make([]float64, 10, 10)
	y[0] = 1
	y1 := mat64.NewDense(10,1, y)

	b1 := []*mat64.Dense{x1, y1}

	x = make([]float64, 784, 784)
	x[2] = 1
	x2 := mat64.NewDense(784, 1, x)
	y = make([]float64, 10, 10)
	y[1] = 1
	y2 := mat64.NewDense(10,1, y)

	b2 := []*mat64.Dense{x2, y2}

	x = make([]float64, 784, 784)
	x[3] = 1
	x3 := mat64.NewDense(784, 1, x)
	y = make([]float64, 10, 10)
	y[2] = 1
	y3 := mat64.NewDense(10,1, y)

	b3 := []*mat64.Dense{x3, y3}

	x = make([]float64, 784, 784)
	x[4] = 1
	x4 := mat64.NewDense(784, 1, x)
	y = make([]float64, 10, 10)
	y[3] = 1
	y4 := mat64.NewDense(10,1, y)

	b4 := []*mat64.Dense{x4, y4}

	miniBatchA := [][]*mat64.Dense{b1, b2}
	miniBatchB := [][]*mat64.Dense{b3, b4}
	miniBatches := [][][]*mat64.Dense{miniBatchA, miniBatchB}
	n.data.miniBatches = miniBatches
	n.updateMiniBatches()

	fmt.Println()

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


	errorMatrix := mat64.NewDense(len(bias1FromPy), 1, nil)

	bias1FromPyDense := mat64.NewDense(len(bias1FromPy), 1, bias1FromPy)
	errorMatrix.Sub(bias1FromPyDense, n.biases[1])

	if mat64.Norm(errorMatrix, 2) > 1e-20 {
		t.Errorf("Error norm exceeding threshold")
	}

	errorMatrix = mat64.NewDense(len(bias0FromPy), 1, nil)
	bias0FromPyDense := mat64.NewDense(len(bias0FromPy), 1, bias0FromPy)
	errorMatrix.Sub(bias0FromPyDense, n.biases[0])

	if mat64.Norm(errorMatrix, 2) > 1e-20 {
		t.Errorf("Error norm exceeding threshold")
	}

	var weights []float64
	for idx1 := range weights1FromPy {
		weights = mat64.Col(nil, idx1, n.weights[1])
		for idx2 := range weights1FromPy[idx1] {
			if math.Abs(weights[idx2] - weights1FromPy[idx1][idx2]) > 1e-8 {
				t.Errorf("Not equal enough:", weights[idx2], weights1FromPy[idx1][idx2])
			}
		}
	}

	for idx1 := range weights0FromPy {
		weights = mat64.Col(nil, idx1, n.weights[0])
		for idx2 := range weights0FromPy[idx1] {
			if math.Abs(weights[idx2] - weights0FromPy[idx1][idx2]) > 1e-8 {
				t.Errorf("Not equal enough:", weights[idx2], weights0FromPy[idx1][idx2])
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


