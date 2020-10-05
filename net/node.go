package net

import (
	"neatgo/activation"
)

func NewNode(id int64, activationFn activation.Fn) *Node {
	return &Node{
		id:           id,
		activationFn: activationFn,
	}
}

type Node struct {
	id           int64
	activationFn activation.Fn
}

func (n *Node) ID() int64 {
	return n.id
}

func (n *Node) Activate(inputs []float64, weights []float64) float64 {
	sum := 0.0
	for i, input := range inputs {
		sum += input * weights[i]
	}

	return n.activationFn(sum)
}

func (n *Node) GetActivationFn() activation.Fn {
	return n.activationFn
}

func (n *Node) Copy() *Node {
	return &Node{
		id:           n.id,
		activationFn: n.activationFn,
	}
}
