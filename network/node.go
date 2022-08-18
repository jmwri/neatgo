package network

type NodeType string

const (
	Hidden NodeType = "hidden"
	Bias            = "bias"
	Input           = "input"
	Output          = "output"
)

func NewNode(id int, nodeType NodeType, bias float64, activationFn ActivationFunctionName) Node {
	return Node{
		ID:           id,
		Type:         nodeType,
		Bias:         bias,
		ActivationFn: activationFn,
	}
}

type Node struct {
	ID           int
	Type         NodeType
	Bias         float64
	ActivationFn ActivationFunctionName
}
