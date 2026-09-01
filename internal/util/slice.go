package util

import "math/rand/v2"

// RemoveSliceIndex removes the element at index i. The order of the remaining
// elements is not preserved.
func RemoveSliceIndex[T any](s []T, i int) []T {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

// RemoveSliceIndexOrdered removes the element at index i, preserving the order
// of the remaining elements.
func RemoveSliceIndexOrdered[T any](s []T, i int) []T {
	return append(s[:i], s[i+1:]...)
}

func InSlice[T comparable](s []T, search T) bool {
	for _, v := range s {
		if v == search {
			return true
		}
	}
	return false
}

// RandSliceElement returns a random element of s. It panics if s is empty.
func RandSliceElement[T any](rng *rand.Rand, s []T) T {
	if len(s) == 0 {
		panic("cannot take a random element of an empty slice")
	}
	if len(s) == 1 {
		return s[0]
	}
	return s[rng.IntN(len(s))]
}
