package util

func RemoveSliceIndex[T any](s []T, i int) []T {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

func InSlice[T comparable](s []T, search T) bool {
	for _, v := range s {
		if v == search {
			return true
		}
	}
	return false
}

func RandSliceElement[T comparable](s []T) T {
	if len(s) == 1 {
		return s[0]
	}
	choice := IntBetween(0, len(s)-1)
	return s[choice]
}
