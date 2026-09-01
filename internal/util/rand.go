package util

import (
	"math/rand/v2"
)

// FloatBetween returns a value in [min, max).
func FloatBetween(rng *rand.Rand, min, max float64) float64 {
	return min + rng.Float64()*(max-min)
}

// Chance reports whether an event with probability p occurs.
func Chance(rng *rand.Rand, p float64) bool {
	if p <= 0 {
		return false
	}
	if p >= 1 {
		return true
	}
	return rng.Float64() < p
}

// Gaussian returns a sample from the standard normal distribution.
//
// The standard library's ziggurat implementation is several times faster than
// the Marsaglia polar method: no rejection loop, and no log or square root on
// the common path. Weight and bias perturbation calls this for nearly every
// gene of every offspring, so it sits squarely in the hot path of a generation.
func Gaussian(rng *rand.Rand) float64 {
	return rng.NormFloat64()
}

// Clamp constrains v to the inclusive range [min, max].
func Clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
