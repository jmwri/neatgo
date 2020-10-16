package activation

import "math/rand"

func RandFn() Fn {
	return FnAll[rand.Intn(len(FnAll))]
}

func RandFnFromOpts(opts []Fn) Fn {
	return opts[rand.Intn(len(opts))]
}
