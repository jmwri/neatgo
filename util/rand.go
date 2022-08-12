package util

import (
	"math"
	"math/rand"
)

type RandFloatProvider func(min, max float64) float64

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

func RandomGaussian() float64 {
	var x1, x2 float64
	w := 10.0
	for w >= 1 {
		x1 = FloatBetween(-1, 1)
		x2 = FloatBetween(-1, 1)
		w = x1*x1 + x2*x2
	}
	w = math.Sqrt((-2 * math.Log(w)) / w)
	return x1 * w
}
