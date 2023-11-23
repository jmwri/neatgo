package activation

func init() {
	DefaultRegistry.Set(None)
}

var None = Wrap("none", func(x float64) float64 {
	return x
})
