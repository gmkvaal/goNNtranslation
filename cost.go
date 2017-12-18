package network

import "math"

func xEntropyCost(a, y []float64) float64 {
	var sum float64
	for idx := range a {
		switch 1-a[idx] {
		case 0:
			sum += 0
		default:
			sum += -y[idx]*math.Log(a[idx]) - (1 - y[idx])*math.Log(1-a[idx])
			}
	}

	return sum
}

func (nf NetworkFormat) totalCost(inputData, outputData [][]float64) float64 {
	var cost float64
	var a []float64
	N := float64(len(outputData))

	for idx := range outputData {
		a = nf.forwardFeed(inputData[idx])
		cost += xEntropyCost(a, outputData[idx])
	}

	cost = cost / (2*N)

	var sumWeights float64
	for k := 0; k < len(nf.Sizes)-1; k++ {
		for j := 0; j < nf.Sizes[k+1]; j++ {
			for i := 0; i < nf.Sizes[k]; i++ {
				sumWeights += nf.weights[k][j][i]*nf.weights[k][j][i]
			}
		}
	}

	sumWeights = sumWeights * nf.hp.lambda/(2*N)
	cost += sumWeights

	return cost
}



