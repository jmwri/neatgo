package feedforward

import (
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/util"
)

type nodeType string

const (
	input  nodeType = "input"
	hidden nodeType = "hidden"
	output nodeType = "output"
)

type Node struct {
	id    int64
	actFn activation.Fn
	bias  float64
	value float64
}

func (n Node) ID() int64 {
	return n.id
}

func (n Node) Value() float64 {
	return n.value
}

func (n Node) Activate(inputs ...float64) Node {
	sum := util.Sum(inputs...)
	n.value = n.actFn.Run(sum + n.bias)
	return n
}
