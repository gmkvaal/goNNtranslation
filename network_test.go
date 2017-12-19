package network

import (
	"testing"

	//plr "github.com/gmkvaal/pythonlistreader"
	"fmt"
	"github.com/gonum/matrix/mat64"
)

/*
// TestForwardFeed tests the forward feed algorithm
// by initiating with zero-weights and biases, hence making
// all z's zero and thus all activations 1/2 (given sigmoids)
func TestForwardFeed(t *testing.T) {
	nf = initNetworkForTesting()
	nf.weights = nf.cubicMatrix(zeroFunc())
	nf.biases = nf.squareMatrix(zeroFunc())
	l := len(nf.Sizes) - 1

	x := make([]float64, 784, 784)
	y := []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	assert.Equal(t, nf.forwardFeed(x, l), y)
}

func TestOutputError(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.Sizes) - 1

	y := []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	yNeg := []float64{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	nf.outputError(y, l)
	assert.Equal(t, nf.delta[l-1], yNeg)
}

func TestBackPropShort(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.Sizes) - 1

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())
	nf.delta[l-1] = []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	nablaB[l-1] = nf.delta[l-1]
	nablaW[l-1] = vectorMatrixProduct(nablaW[l-1], nf.delta[l-1], nf.activations[l-1])
	nablaW, nablaB = nf.backPropError(nablaW, nablaB, l)
	nablaB[l-1] = nf.delta[l-1]
	nablaW[l-1] = vectorMatrixProduct(nablaW[l-1], nf.delta[l-1], nf.activations[l-1])
	nablaW, nablaB = nf.backPropError(nablaW, nablaB, l)
}


func TestOutputGradients(t *testing.T) {
	nf = initNetworkForTesting()
	l := len(nf.Sizes) - 1

	nablaW := nf.cubicMatrix(zeroFunc())
	nablaB := nf.squareMatrix(zeroFunc())

	nf.outputGradients(nablaW, nablaB, l)
}



// TestBackProp tests the back propagation algorithm
// by running through two mini batches of length two,
// and then comparing the weights and biases with the
// validated solver https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src/network2
// Weights and biases are for the purpose of creating
// a deterministic result initiated as 1's.
func TestBackProp(t *testing.T) {
	nf := networkFormat{Sizes: []int{784, 30, 10}}

	nf.weights = nf.cubicMatrix(oneFunc())
	nf.biases = nf.squareMatrix(oneFunc())
	nf.delta = nf.squareMatrix(zeroFunc())
	nf.z = nf.squareMatrix(zeroFunc())
	nf.activations = nf.squareMatrixFull(zeroFunc())

	nf.miniBatchSize = 2
	nf.n = 4
	nf.hp.eta = 1
	nf.hp.lambda = 5.0

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
	nf.data.miniBatches = miniBatches
	nf.updateMiniBatches()


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

	for idx := range bias1FromPy {
		if bias1FromPy[idx] - nf.biases[1][idx] > 1e-7 {
			t.Error("not equal", bias1FromPy[idx], nf.biases[1][idx])
		}
	}

	for idx := range bias0FromPy {
		if bias0FromPy[idx] - nf.biases[0][idx] > 1e-7 {
			t.Error("not equal", bias0FromPy[idx], nf.biases[0][idx])
		}
	}

	for idx1 := range weights1FromPy {
		for idx2 := range weights1FromPy[idx1] {
			if weights1FromPy[idx1][idx2] - nf.weights[1][idx1][idx2] > 1e-7 {
				t.Error("not equal", weights1FromPy[idx1][idx2], nf.weights[1][idx1][idx2])
			}
		}
	}

	for idx1 := range weights0FromPy {
		for idx2 := range weights0FromPy[idx1] {
			if weights0FromPy[idx1][idx2] - nf.weights[0][idx1][idx2] > 1e-7 {
				t.Error("not equal", weights0FromPy[idx1][idx2], nf.weights[0][idx1][idx2])
			}
		}
	}
}


*/

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

func TestIt(t *testing.T) {
	fmt.Println()
	nf := NetworkFormat{Sizes: []int{2, 3, 1}}
	nf.InitNetwork()
	nf.InitNetworkMethods(OutputErrorXEntropy, SigmoidActivation)
	nf.data.loadData()
	nf.n = 4
	nf.hp.eta = 0.5
	nf.hp.lambda = 0.001


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


	nf.weights = []*mat64.Dense{w1, w2}

	b1 := mat64.NewDense(3, 1, nil)
	b1.Set(0, 0, 1.47931576)
	b1.Set(1, 0, 5.76116679)
	b1.Set(2, 0, 4.77665241)

	b2 := mat64.NewDense(1,1,nil)
	b2.Set(0,0,-7.41086319)

	nf.biases = []*mat64.Dense{b1, b2}

	//fmt.Println(nf.weights[1].RawMatrix().Data)


	//nf.miniBatchGenerator(0, 4, 1, true)

	/*
	for i := 0; i < 1000; i++ {
		nf.miniBatchGenerator(0, 4, 1, false)
		nf.updateMiniBatches()
	}
	*/
	var y *mat64.Dense
	y = nf.ForwardFeedRapid(mat64.NewDense(2,1, []float64{0,0}))
	fmt.Println(y.RawMatrix().Data)
	y = nf.ForwardFeedRapid(mat64.NewDense(2,1, []float64{1,0}))
	fmt.Println(y.RawMatrix().Data)
	y = nf.ForwardFeedRapid(mat64.NewDense(2,1, []float64{0,1}))
	fmt.Println(y.RawMatrix().Data)
	y = nf.ForwardFeedRapid(mat64.NewDense(2,1, []float64{1,1}))
	fmt.Println(y.RawMatrix().Data)

	fmt.Println()




	/*
	for factor := 0; factor < 100; factor++ {
		nf.hp.lambda = float64(factor) * 0.01

		for i := 0; i < 500; i++ {
			nf.miniBatchGenerator(0, 4, 4, true)
			nf.updateMiniBatches()
		}

		fmt.Println(nf.hp.lambda)
		y = nf.forwardFeed([]float64{0, 0}, 2)
		fmt.Println(y, 1)
		y = nf.forwardFeed([]float64{1, 0}, 2)
		fmt.Println(y, 1)
		y = nf.forwardFeed([]float64{0, 1}, 2)
		fmt.Println(y, 1)
		y = nf.forwardFeed([]float64{1, 1}, 2)
		fmt.Println(y, 0)

		fmt.Println()

		//fmt.Println(nf.weights)

		//nf.TrainNetwork(0, 10, 4, 0.1, 0.1, false)

	}
	*/
}
