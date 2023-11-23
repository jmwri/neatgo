package util

func Sum[T float64 | int64](x ...T) T {
	var tot T
	for i, val := range x {
		if i == 0 {
			tot = val
			continue
		}
		tot += val
	}
	return tot
}
