package activation

import "math/rand"

func RandFn() Fn {
	return FnAll[rand.Intn(len(FnAll))]
}
