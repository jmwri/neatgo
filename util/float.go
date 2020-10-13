package util

import "math/rand"

func RandFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

type ReduceFn func(x, y float64) float64

func ReduceFloat(fn ReduceFn, sequence []float64, init float64) float64 {
	i := 0
	value := init
	if value == 0 {
		value = sequence[i]
		i++
	}

	for ; i < len(sequence); i++ {
		value = fn(value, sequence[i])
	}

	return value
}

func MultiplyFloat(x, y float64) float64 {
	return x * y
}

func SumFloat(x, y float64) float64 {
	return x + y
}
