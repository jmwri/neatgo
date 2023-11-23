package util

import "math/rand"

func FloatBetween(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func IntBetween(min, max int) int {
	if max < min {
		panic("min must be smaller than max")
	}
	if min == max {
		return min
	}
	return min + rand.Intn((max+1)-min)
}
