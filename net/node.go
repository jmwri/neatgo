package net

import (
	"neatgo/activation"
	"neatgo/aggregation"
)

func NewNode(id int64, bias float64, activationFn activation.Fn, aggregationFn aggregation.Fn) *Node {
	return &Node{
		id:            id,
		bias:          bias,
		activationFn:  activationFn,
		aggregationFn: aggregationFn,
		activation:    0,
	}
}

type Node struct {
	id            int64
	bias          float64
	activationFn  activation.Fn
	aggregationFn aggregation.Fn
	activation    float64
}

func (n *Node) ID() int64 {
	return n.id
}

func (n *Node) Bias() float64 {
	return n.bias
}

func (n *Node) SetBias(b float64) {
	n.bias = b
}

func (n *Node) Activate(inputs []float64, weights []float64) float64 {
	values := make([]float64, len(inputs))
	for i, input := range inputs {
		values[i] = input * weights[i]
	}
	agg := n.aggregationFn(values)
	return n.activationFn(n.bias + agg)
}

func (n *Node) ActivationFn() activation.Fn {
	return n.activationFn
}

func (n *Node) SetActivationFn(fn activation.Fn) {
	n.activationFn = fn
}

func (n *Node) AggregationFn() aggregation.Fn {
	return n.aggregationFn
}

func (n *Node) SetAggregationFn(fn aggregation.Fn) {
	n.aggregationFn = fn
}

func (n *Node) Activation() float64 {
	return n.activation
}

func (n *Node) SetActivation(v float64) {
	n.activation = v
}

func (n *Node) Copy() *Node {
	return &Node{
		id:            n.id,
		bias:          n.bias,
		activationFn:  n.activationFn,
		aggregationFn: n.aggregationFn,
		activation:    n.activation,
	}
}
