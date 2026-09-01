package neat

import "math/rand/v2"

// Rand is the random source driving a run.
//
// Evolution is stochastic, so nothing about a run is repeatable unless every
// random choice comes from a source the caller controls. Reaching for the
// package-level math/rand functions instead leaves the sequence at the mercy of
// process-wide state, which makes a promising result impossible to reproduce
// and a rare crash impossible to bisect.
//
// A Rand is not safe for concurrent use. Reproduction runs on a single
// goroutine and uses the population's own source; fitness evaluation runs in
// parallel but never touches it.
type Rand = rand.Rand

// NewRand returns a random source for the given seed. The same seed always
// produces the same sequence.
func NewRand(seed uint64) *Rand {
	// PCG is cheap to construct and to seed, which matters because a source
	// may be built per run rather than once per process.
	return rand.New(rand.NewPCG(seed, seed^pcgStreamOffset))
}

// pcgStreamOffset separates the two halves of a PCG seed so that a seed and
// its derived stream are not trivially related.
const pcgStreamOffset = 0x9E3779B97F4A7C15

// generationSeed derives a generation's random stream from the run's seed.
//
// Deriving it rather than letting one stream run on across generations means a
// generation's randomness depends only on the seed and the generation number.
// That is what lets a run be picked up from a snapshot and continue exactly as
// it would have: an uninterrupted run and a resumed one reach the same place.
func generationSeed(seed uint64, generation int) uint64 {
	// splitmix64 finaliser, which scrambles even closely-related inputs.
	x := seed + uint64(generation)*pcgStreamOffset
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	return x
}

// RandomSeed returns a seed drawn from the process-wide source, for when the
// caller does not care which run they get but still wants to record it.
func RandomSeed() uint64 {
	return rand.Uint64()
}
