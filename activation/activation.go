package activation

import "math"

type Fn func(x float64) float64

var FnAll = []Fn{
	Nil,
	Sigmoid,
	Tanh,
	Sin,
	Gauss,
	Relu,
	Softplus,
	Clamped,
	Inv,
	Log,
	Exp,
	Abs,
	Hat,
	Square,
	Cube,
}

func Nil(x float64) float64 {
	return x
}

func Sigmoid(x float64) float64 {
	x = math.Max(-60.0, math.Min(60.0, 5.0*x))
	return 1.0 / (1.0 + math.Exp(-x))
}

func Tanh(x float64) float64 {
	x = math.Max(-60.0, math.Min(60.0, 2.5*x))
	return math.Tanh(x)
}

func Sin(x float64) float64 {
	x = math.Max(-60.0, math.Min(60.0, 5.0*x))
	return math.Sin(x)
}

func Gauss(x float64) float64 {
	x = math.Max(-3.4, math.Min(3.4, x))
	return math.Exp(-5.0 * math.Pow(x, 2))
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func Softplus(x float64) float64 {
	x = math.Max(-60.0, math.Min(60.0, 5.0*x))
	return 0.2 * math.Log(1+math.Exp(x))
}

func Clamped(x float64) float64 {
	return math.Max(-1.0, math.Min(1.0, x))
}

func Inv(x float64) float64 {
	if x == 0 {
		return x
	}
	return 1.0 / x
}

func Log(x float64) float64 {
	x = math.Max(1e-7, x)
	return math.Log(x)
}

func Exp(x float64) float64 {
	x = math.Max(-60.0, math.Min(60.0, x))
	return math.Exp(x)
}

func Abs(x float64) float64 {
	return math.Abs(x)
}

func Hat(x float64) float64 {
	return math.Max(0.0, 1.0-math.Abs(x))
}

func Square(x float64) float64 {
	return math.Pow(x, 2)
}

func Cube(x float64) float64 {
	return math.Pow(x, 3)
}
