package network

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