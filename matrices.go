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