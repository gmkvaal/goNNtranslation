package network

import "github.com/gonum/matrix/mat64"

// squareMatrix creates a square matrix with entries from input function
func (nf NetworkFormat) squareMatrix(input func(int) float64) [][]float64 {
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
func (nf NetworkFormat) squareMatrixFull(input func(int) float64) [][]float64 {
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
func (nf NetworkFormat) cubicMatrix(input func(int) float64) [][][]float64 {
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

// sliceWithGonumVectors creates a slice containing gonum vectors. The length of
// each respective vector is given in the slice `sizes`, and the vector
// entries are given by the function `input`
func sliceWithGonumVectors(start int, sizes []int, input func(int) float64) []*mat64.Vector {
	b := make([]*mat64.Vector, len(sizes[start:]))

	for k := range b {
		b[k] = mat64.NewVector(sizes[k+start], nil)

		for j := 0; j < sizes[k+start]; j++ {
			b[k].SetVec(j, input(sizes[k+start]))
		}
	}

	return b
}

// sliceWithGonumMatrices creates a slice containing gonum dense (matrix).
func sliceWithGonumMatrices(sizes []int, input func(int) float64) []*mat64.Dense {
	w := make([]*mat64.Dense, len(sizes[1:]))

	for k := range w {
		w[k] = mat64.NewDense(sizes[k], sizes[k+1], nil)

		for j := 0; j < sizes[k+1]; j++ {
			for i := 0; i < sizes[k]; i++ {
				w[k].Set(i, j, input(sizes[k+1]))
			}
		}
	}

	return w
}


func swgm(start int, sizes []int, input func(int) float64) []*mat64.Dense {
	w := make([]*mat64.Dense, len(sizes[start:]))

	for k := range w {
		w[k] = mat64.NewDense(sizes[k+start], 1, nil)

		for j := 0; j < 1; j++ {
			for i := 0; i < sizes[k+start]; i++ {
				w[k].Set(i, j, input(sizes[k+1]))
			}
		}
	}

	return w
}

// sliceWithGonumDense creates a slices containing gonum denses (matrices). The number of denses contained is
// lenSlice. The denses at position k has number of rows given by rows[k] and number of columns given either by
// col[k] or col (int) for a fixed number of columns (e.g 1 for vectors). The entries of the denses are given by input
func SliceWithGonumDense(lenSlice int, rows []int, cols interface{}, input func(int) float64) []*mat64.Dense {
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