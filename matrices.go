package main

// squareMatrix creates a square matrix with entries from input function
func (nf networkFormat) squareMatrix(input func(int) float64) [][]float64 {
	b := make([][]float64, len(nf.sizes[1:]))

	for k := range b {
		b[k] = make([]float64, nf.sizes[k+1])

		for j := 0; j < nf.sizes[k+1]; j++ {
			b[k][j] = input(nf.sizes[k+1])
		}
	}

	return b
}

// squareMatrix creates a square matrix with entries from input function
func (nf networkFormat) squareMatrixFull(input func(int) float64) [][]float64 {
	b := make([][]float64, len(nf.sizes))

	for k := range b {
		b[k] = make([]float64, nf.sizes[k])

		for j := 0; j < nf.sizes[k]; j++ {
			b[k][j] = input(nf.sizes[k])
		}
	}

	return b
}

// cubicMatrix creates a square matrix with entries from input function
func (nf networkFormat) cubicMatrix(input func(int) float64) [][][]float64 {
	w := make([][][]float64, len(nf.sizes[1:]))

	for k := range w {
		w[k] = make([][]float64, nf.sizes[k+1])

		for j := 0; j < nf.sizes[k+1]; j++ {
			w[k][j] = make([]float64, nf.sizes[k])

			for i := 0; i < nf.sizes[k]; i++ {
				w[k][j][i] = input(nf.sizes[k])
			}
		}
	}

	return w
}