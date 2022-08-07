package network

type NodeType int

const (
	Hidden NodeType = iota
	Bias
	Input
	Output
)

func NewNode(id int, nodeType NodeType, bias float64, activationFn ActivationFunction) Node {
	return Node{
		ID:           id,
		Type:         nodeType,
		Bias:         bias,
		ActivationFn: activationFn,
	}
}

type ActivationFunction func(state float64) float64
type Node struct {
	ID           int
	Type         NodeType
	Bias         float64
	ActivationFn ActivationFunction
}
