package simpletest


type Data struct {
	TrainingInput    [][]float64
	TrainingOutput   [][]float64
	ValidationInput  [][]float64
	ValidationOutput [][]float64
}


func (d *Data) InitTestSlices() {
	tI := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	tO := [][]float64{{1},{1},{1},{0}}
	d.TrainingInput = tI
	d.TrainingOutput = tO
}

