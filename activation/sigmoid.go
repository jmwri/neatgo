package activation

import (
	"math"
)

func init() {
	DefaultRegistry.Set(Sigmoid)
}

var Sigmoid = Wrap("sigmoid", func(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
})
