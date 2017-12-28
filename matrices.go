package network

import "github.com/gonum/matrix/mat64"


// sliceWithGonumDense creates a slices containing gonum denses (matrices). The number of denses contained is
// lenSlice. The denses at position k has number of rows given by rows[k] and number of columns given either by
// col[k] or col (int) for a fixed number of columns (e.g 1 for vectors). The entries of the denses are given by input
func sliceWithGonumDense(lenSlice int, rows []int, cols interface{}, input func(int) float64) []*mat64.Dense {
	w := make([]*mat64.Dense, lenSlice)

	for k := range w {
		switch cols.(type) {
		case int:
			w[k] = mat64.NewDense(rows[k], cols.(int), nil)
			for j := 0; j < cols.(int); j++ {
				for i := 0; i < rows[k]; i++ {
					w[k].Set(i, j, input(rows[k]))
				}
			}
		case []int:
			w[k] = mat64.NewDense(rows[k], cols.([]int)[k], nil)
			for j := 0; j < cols.([]int)[k]; j++ {
				for i := 0; i < rows[k]; i++ {
					w[k].Set(i, j, input(rows[k]))
				}
			}
		}
	}
	return w
}


// squareMatrix creates a square matrix with entries from input function
func (nf Network) squareMatrix(input func(int) float64) [][]float64 {
	b := make([][]float64, len(nf.Sizes[1:]))

	for k := range b {
		b[k] = make([]float64, nf.Sizes[k+1])

		for j := 0; j < nf.Sizes[k+1]; j++ {
			b[k][j] = input(nf.Sizes[k+1])
		}
	}

	return b
}

// squareMatrix creates a square matrix with entries from input function
func (nf Network) squareMatrixFull(input func(int) float64) [][]float64 {
	b := make([][]float64, len(nf.Sizes))

	for k := range b {
		b[k] = make([]float64, nf.Sizes[k])

		for j := 0; j < nf.Sizes[k]; j++ {
			b[k][j] = input(nf.Sizes[k])
		}
	}

	return b
}

// cubicMatrix creates a square matrix with entries from input function
func (nf Network) cubicMatrix(input func(int) float64) [][][]float64 {
	w := make([][][]float64, len(nf.Sizes[1:]))

	for k := range w {
		w[k] = make([][]float64, nf.Sizes[k+1])

		for j := 0; j < nf.Sizes[k+1]; j++ {
			w[k][j] = make([]float64, nf.Sizes[k])

			for i := 0; i < nf.Sizes[k]; i++ {
				w[k][j][i] = input(nf.Sizes[k])
			}
		}
	}

	return w
}