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
	ActivationRegistry.Set(Identity, IdentityFn)
	ActivationRegistry.Set(Sigmoid, SigmoidFn)
	ActivationRegistry.Set(Tanh, TanhFn)
	ActivationRegistry.Set(Sin, SinFn)
	ActivationRegistry.Set(Gauss, GaussFn)
	ActivationRegistry.Set(Relu, ReluFn)
	ActivationRegistry.Set(Elu, EluFn)
	ActivationRegistry.Set(Lelu, LeluFn)
	ActivationRegistry.Set(Selu, SeluFn)
	ActivationRegistry.Set(SoftPlus, SoftPlusFn)
	ActivationRegistry.Set(Clamped, ClampedFn)
	ActivationRegistry.Set(Inv, InvFn)
	ActivationRegistry.Set(Log, LogFn)
	ActivationRegistry.Set(Exp, ExpFn)
	ActivationRegistry.Set(Abs, AbsFn)
	ActivationRegistry.Set(Hat, HatFn)
	ActivationRegistry.Set(Square, SquareFn)
	ActivationRegistry.Set(Cube, CubeFn)
}

func RandomActivationFunction(choices ...ActivationFunctionName) ActivationFunction {
	if len(choices) == 0 {
		choices = ActivationRegistry.Names()
	}
	randomSelection := util.IntBetween(0, len(choices))
	return ActivationRegistry.Get(choices[randomSelection])
}

const (
	NoActivation ActivationFunctionName = "no-activation"
	Identity                            = "identity"
	Sigmoid                             = "sigmoid"
	Tanh                                = "tanh"
	Sin                                 = "sin"
	Gauss                               = "gauss"
	Relu                                = "relu"
	Elu                                 = "elu"
	Lelu                                = "lelu"
	Selu                                = "selu"
	SoftPlus                            = "softplus"
	Clamped                             = "clamped"
	Inv                                 = "inv"
	Log                                 = "log"
	Exp                                 = "exp"
	Abs                                 = "abs"
	Hat                                 = "hat"
	Square                              = "square"
	Cube                                = "cube"
)

func NoActivationFn(x float64) float64 {
	return IdentityFn(x)
}

func IdentityFn(x float64) float64 {
	return x
}

func SigmoidFn(x float64) float64 {
	x = math.Max(-60, math.Min(60, 5*x))
	return 1.0 / (1.0 + math.Exp(-x))
}

func TanhFn(x float64) float64 {
	x = math.Max(-60, math.Min(60, 2.5*x))
	return math.Tanh(x)
}

func SinFn(x float64) float64 {
	x = math.Max(-60, math.Min(60, 5*x))
	return math.Sin(x)
}

func GaussFn(x float64) float64 {
	x = math.Max(-3.4, math.Min(3.4, x))
	return math.Exp(math.Pow(-5*x, 2))
}

func ReluFn(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func EluFn(x float64) float64 {
	if x > 0 {
		return x
	}
	return math.Exp(x) - 1
}

func LeluFn(x float64) float64 {
	if x > 0 {
		return x
	}
	leaky := .005
	return leaky * x
}

func SeluFn(x float64) float64 {
	lam := 1.0507009873554804934193349852946
	alpha := 1.6732632423543772848170429916717
	if x > 0 {
		return lam * x
	}
	return lam * alpha * (math.Exp(x) - 1)
}

func SoftPlusFn(x float64) float64 {
	x = math.Max(-60, math.Min(60, 5*x))
	return .2 * math.Log(1+math.Exp(x))
}

func ClampedFn(x float64) float64 {
	return math.Max(-1, math.Min(1, x))
}

func InvFn(x float64) float64 {
	if x == 0 {
		return 0
	}
	return 1 / x
}

func LogFn(x float64) float64 {
	x = math.Max(1e-7, x)
	return math.Log(x)
}

func ExpFn(x float64) float64 {
	x = math.Max(-60, math.Min(60, x))
	return math.Exp(x)
}

func AbsFn(x float64) float64 {
	return math.Abs(x)
}

func HatFn(x float64) float64 {
	return math.Max(0, 1-math.Abs(x))
}

func SquareFn(x float64) float64 {
	return math.Pow(x, 2)
}

func CubeFn(x float64) float64 {
	return math.Pow(x, 3)
}
