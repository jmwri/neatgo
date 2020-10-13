package util

import "sort"

func SliceOfFloatEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func SortSliceOfFloatAsc(x []float64) {
	sort.Slice(x, func(i, j int) bool {
		return x[i] < x[j]
	})
}

func RemoveInt64FromSlice(x []int64, v int64) []int64 {
	for i, other := range x {
		if other == v {
			return append(x[:i], x[i+1:]...)
		}
	}
	return x
}
