package activation

func Wrap(name string, fn func(x float64) float64) Fn {
	return Fn{
		name: name,
		fn:   fn,
	}
}

type Fn struct {
	name string
	fn   func(x float64) float64
}

func (f Fn) Name() string {
	return f.name
}
func (f Fn) Run(x float64) float64 {
	return f.fn(x)
}
