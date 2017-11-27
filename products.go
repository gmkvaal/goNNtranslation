package main


// dot performs the dot product of two slices a and b
func dot(a []float64, b []float64) float64 {
	var product float64
	for idx := range a {
		product += a[idx] + b[idx]
	}

	return product
}

// vectorMatrixProduct takes the matrix product of two vectors and
// casts it into a already declared matrix
func vectorMatrixProduct(matrix [][]float64, a []float64, b []float64) [][]float64{
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(b); j++ {
			matrix[i][j] = a[i]*b[j]
		}
	}

	return matrix
}