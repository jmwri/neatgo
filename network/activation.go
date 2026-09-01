package network

import (
	"math"
	"math/rand/v2"
	"sync"
	"sync/atomic"

	"github.com/jmwri/neatgo/v2/internal/util"
)

type ActivationFunction func(x float64) float64

type ActivationFunctionName string

// activationRegistry holds the activation functions available to genomes.
// Reads are lock-free: the function table is swapped atomically on write so
// that Get can be called from evaluation hot loops without contention.
type activationRegistry struct {
	mu        sync.Mutex
	functions atomic.Pointer[map[ActivationFunctionName]ActivationFunction]
	names     []ActivationFunctionName
}

func (r *activationRegistry) Set(n ActivationFunctionName, fn ActivationFunction) {
	r.mu.Lock()
	defer r.mu.Unlock()

	current := r.functions.Load()
	next := make(map[ActivationFunctionName]ActivationFunction, len(*current)+1)
	for name, existing := range *current {
		next[name] = existing
	}

	_, known := next[n]
	if fn == nil {
		delete(next, n)
		if known {
			for i, name := range r.names {
				if name == n {
					r.names = append(r.names[:i], r.names[i+1:]...)
					break
				}
			}
		}
	} else {
		next[n] = fn
		if !known {
			// Only track the name on first registration, otherwise overriding a
			// function would list it twice and bias random selection.
			r.names = append(r.names, n)
		}
	}

	r.functions.Store(&next)
}

func (r *activationRegistry) Get(n ActivationFunctionName) ActivationFunction {
	return (*r.functions.Load())[n]
}

// Names returns a copy of the registered activation function names.
func (r *activationRegistry) Names() []ActivationFunctionName {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]ActivationFunctionName, len(r.names))
	copy(names, r.names)
	return names
}

var ActivationRegistry = newActivationRegistry()

func newActivationRegistry() *activationRegistry {
	r := &activationRegistry{
		names: make([]ActivationFunctionName, 0),
	}
	empty := make(map[ActivationFunctionName]ActivationFunction)
	r.functions.Store(&empty)

	// Registered here rather than in an init function so that the registry is
	// never observable in an empty state.
	r.Set(NoActivation, NoActivationFn)
	r.Set(Identity, IdentityFn)
	r.Set(Sigmoid, SigmoidFn)
	r.Set(Tanh, TanhFn)
	r.Set(Sin, SinFn)
	r.Set(Gauss, GaussFn)
	r.Set(Relu, ReluFn)
	r.Set(Elu, EluFn)
	r.Set(Lelu, LeluFn)
	r.Set(Selu, SeluFn)
	r.Set(SoftPlus, SoftPlusFn)
	r.Set(Clamped, ClampedFn)
	r.Set(Inv, InvFn)
	r.Set(Log, LogFn)
	r.Set(Exp, ExpFn)
	r.Set(Abs, AbsFn)
	r.Set(Hat, HatFn)
	r.Set(Square, SquareFn)
	r.Set(Cube, CubeFn)

	return r
}

// RandomActivationFunction picks one of the given activation functions, or one
// of every registered function when no choices are given.
func RandomActivationFunction(rng *rand.Rand, choices ...ActivationFunctionName) ActivationFunctionName {
	if len(choices) == 0 {
		choices = ActivationRegistry.Names()
	}
	return util.RandSliceElement(rng, choices)
}

const (
	NoActivation ActivationFunctionName = "no-activation"
	Identity     ActivationFunctionName = "identity"
	Sigmoid      ActivationFunctionName = "sigmoid"
	Tanh         ActivationFunctionName = "tanh"
	Sin          ActivationFunctionName = "sin"
	Gauss        ActivationFunctionName = "gauss"
	Relu         ActivationFunctionName = "relu"
	Elu          ActivationFunctionName = "elu"
	Lelu         ActivationFunctionName = "lelu"
	Selu         ActivationFunctionName = "selu"
	SoftPlus     ActivationFunctionName = "softplus"
	Clamped      ActivationFunctionName = "clamped"
	Inv          ActivationFunctionName = "inv"
	Log          ActivationFunctionName = "log"
	Exp          ActivationFunctionName = "exp"
	Abs          ActivationFunctionName = "abs"
	Hat          ActivationFunctionName = "hat"
	Square       ActivationFunctionName = "square"
	Cube         ActivationFunctionName = "cube"
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
	// exp(-5x^2). Note the sign is inside the exponent, not on the squared term.
	return math.Exp(-5 * x * x)
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
