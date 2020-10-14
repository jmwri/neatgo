package aggregation

import (
	"github.com/jmwri/neatgo/util"
	"math"
)

type Fn func(x []float64) float64

var FnAll = []Fn{
	Product,
	Sum,
	Max,
	Min,
	MaxAbs,
	Mean,
	Median,
}

func Product(x []float64) float64 {
	return util.ReduceFloat(util.MultiplyFloat, x, 1)
}

func Sum(x []float64) float64 {
	return util.ReduceFloat(util.SumFloat, x, 0)
}

func Max(x []float64) float64 {
	var max float64
	setMax := false
	for _, f := range x {
		if !setMax || f > max {
			max = f
			setMax = true
		}
	}
	return max
}

func Min(x []float64) float64 {
	var min float64
	setMin := false
	for _, f := range x {
		if !setMin || f < min {
			min = f
			setMin = true
		}
	}
	return min
}

func MaxAbs(x []float64) float64 {
	var max float64
	setMax := false
	for _, f := range x {
		absF := math.Abs(f)
		if !setMax || absF > max {
			max = absF
			setMax = true
		}
	}
	return max
}

func Mean(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	v := x[0]
	for i := 1; i < len(x); i++ {
		v += x[i]
	}
	return v / float64(len(x))
}

func Median(x []float64) float64 {
	xLen := len(x)
	if xLen <= 2 {
		return Mean(x)
	}

	cp := make([]float64, len(x))
	for i := range x {
		cp[i] = x[i]
	}
	util.SortSliceOfFloatAsc(cp)

	midIndex := math.Floor(float64(xLen) / 2)
	if xLen%2 == 1 {
		return cp[int(midIndex)]
	}

	return Mean([]float64{midIndex - 1, midIndex})
}

func Variance(x []float64) float64 {
	mean := Mean(x)
	values := make([]float64, 0)
	for _, xv := range x {
		values = append(values, math.Pow(xv-mean, 2))
	}

	return Sum(values) / float64(len(x))
}

func Stdev(x []float64) float64 {
	return math.Sqrt(Variance(x))
}
