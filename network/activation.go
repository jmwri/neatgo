package network

import (
	"github.com/jmwri/neatgo/util"
	"math"
	"sync"
)

type ActivationFunction func(x float64) float64

type ActivationFunctionName string

type activationRegistry struct {
	mu        sync.Mutex
	functions map[ActivationFunctionName]ActivationFunction
	names     []ActivationFunctionName
}

func (r *activationRegistry) Set(n ActivationFunctionName, fn ActivationFunction) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if fn == nil {
		delete(r.functions, n)
		for i, name := range r.names {
			if name == n {
				r.names = util.RemoveSliceIndex(r.names, i)
			}
		}
	} else {
		r.functions[n] = fn
		r.names = append(r.names, n)
	}
}

func (r *activationRegistry) Get(n ActivationFunctionName) ActivationFunction {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.functions[n]
}

func (r *activationRegistry) Names() []ActivationFunctionName {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.names
}

var ActivationRegistry = &activationRegistry{
	mu:        sync.Mutex{},
	functions: make(map[ActivationFunctionName]ActivationFunction),
}

func init() {
	ActivationRegistry.Set(NoActivation, NoActivationFn)
	ActivationRegistry.Set(Tanh, math.Tan)
}

func RandomActivationFunction() ActivationFunction {
	names := ActivationRegistry.Names()
	randomSelection := util.IntBetween(0, len(names))
	return ActivationRegistry.Get(names[randomSelection])
}

const (
	NoActivation ActivationFunctionName = "no-activation"
	Tanh         ActivationFunctionName = "tanh"
)

func NoActivationFn(x float64) float64 {
	return x
}
