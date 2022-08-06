package network

type NodeType int

const (
	Hidden NodeType = iota
	Input
	Output
)

type Node struct {
	ID           int
	Type         NodeType
	Bias         float64
	ActivationFn func(state float64) float64
}
