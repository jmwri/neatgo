package net

import (
	"errors"
	"neatgo"
)

var ErrInputLen = errors.New("input length does not match input layer")

type NeuralNetwork interface {
	neatgo.Identifier
	Activate(inputs []float64) ([]float64, error)
}

func NewFeedForward(id int64, biasNodes []BiasNode, inputNodes []InputNode, hiddenNodes []HiddenNode, outputNodes []OutputNode) *FeedForward {
	return &FeedForward{
		id:          id,
		biasNodes:   biasNodes,
		inputNodes:  inputNodes,
		hiddenNodes: hiddenNodes,
		outputNodes: outputNodes,
	}
}

type FeedForward struct {
	id          int64
	biasNodes   []BiasNode
	inputNodes  []InputNode
	hiddenNodes []HiddenNode
	outputNodes []OutputNode
}

func (n *FeedForward) ID() int64 {
	return n.id
}

func (n *FeedForward) Activate(inputs []float64) ([]float64, error) {
	output := make([]float64, len(n.outputNodes))
	if len(inputs) != len(n.inputNodes) {
		return output, ErrInputLen
	}

	// Set the input of each input node and activate them
	for i, inputValue := range inputs {
		n.inputNodes[i].SetInput(inputValue)
		n.inputNodes[i].Activate()
	}

	// Activate each bias node
	for _, biasNode := range n.biasNodes {
		biasNode.Activate()
	}

	// Activate each hidden node
	for _, hiddenNode := range n.hiddenNodes {
		hiddenNode.Activate()
	}

	// Activate each output node and gather output
	for i, outputNode := range n.outputNodes {
		outputNode.Activate()
		output[i] = outputNode.Output()
	}

	return output, nil
}
