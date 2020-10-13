package net

import (
	"errors"
	"fmt"
	"neatgo/activation"
	"neatgo/aggregation"
	"neatgo/util"
)

var ErrInputLen = errors.New("input length does not match input layer")
var ErrShapeNotBigEnough = errors.New("shape must be at least 2")

type NeuralNetwork interface {
	ID() int64
	Activate(inputs []float64) ([]float64, error)
	Copy() NeuralNetwork
	Layers() []Layer
	Connections() []LayerConnections
}

func NewFeedForwardFromDefinition(id int64, layerDefinitions []LayerDefinition) (*FeedForward, error) {
	if len(layerDefinitions) < 2 {
		return nil, ErrShapeNotBigEnough
	}

	// Build up layers based on layerDefinitions
	layers := make([]Layer, len(layerDefinitions))
	var nodeID int64 = 0
	for i := 0; i < len(layerDefinitions); i++ {
		layers[i] = make([]*Node, layerDefinitions[i].NumNodes)
		for ni := 0; ni < layerDefinitions[i].NumNodes; ni++ {
			bias := util.RandFloat(layerDefinitions[i].BiasInitMin, layerDefinitions[i].BiasInitMax)
			activationFn := layerDefinitions[i].ActivationFn
			if activationFn == nil {
				activationFn = activation.Nil
			}
			aggregationFn := layerDefinitions[i].AggregationFn
			if aggregationFn == nil {
				aggregationFn = aggregation.Sum
			}
			layers[i][ni] = NewNode(nodeID, bias, activationFn, aggregationFn)
			nodeID++
		}
	}

	connections := make([]LayerConnections, len(layers)-1)
	// Fully-connect our layers
	var connID int64 = 0
	for i := 0; i < len(layers)-1; i++ {
		connections[i] = make([]*Connection, 0)
		for _, nodeFrom := range layers[i] {
			for _, nodeTo := range layers[i+1] {
				c := NewConnection(connID, 1, nodeFrom.ID(), nodeTo.ID(), true)
				connections[i] = append(connections[i], c)
				connID++
			}
		}
	}

	return NewFeedForward(id, layers, connections), nil
}

func NewFeedForward(id int64, layers []Layer, connections []LayerConnections) *FeedForward {
	return &FeedForward{
		id:          id,
		layers:      layers,
		connections: connections,
	}
}

type FeedForward struct {
	id          int64
	layers      []Layer
	connections []LayerConnections
}

func (n *FeedForward) ID() int64 {
	return n.id
}

func (n *FeedForward) Layers() []Layer {
	return n.layers
}

func (n *FeedForward) Connections() []LayerConnections {
	return n.connections
}

func (n *FeedForward) Activate(inputs []float64) ([]float64, error) {
	outputLayer := n.layers[len(n.layers)-1]
	output := make([]float64, len(outputLayer))

	if len(inputs) != len(n.layers[0]) {
		return output, fmt.Errorf("failed to handle layer: %w", ErrInputLen)
	}

	// We're going to update these vars as we process each layer
	// Initialise them with our inputs
	layerInputs := make(map[int64][]float64)
	layerWeights := make(map[int64][]float64)
	for i, val := range inputs {
		inputNode := n.layers[0][i]
		layerInputs[inputNode.ID()] = []float64{val}
		layerWeights[inputNode.ID()] = []float64{1}
	}

	// For each layer...
	for layerI, layer := range n.layers {
		// Make sure we have enough inputs + weights
		if len(layerInputs) != len(layer) || len(layerWeights) != len(layer) {
			return output, fmt.Errorf("failed to handle layer %d: %w", layerI, ErrInputLen)
		}

		// Loop over each node and pass in our inputs
		// Gather each nodes outputs
		nodeOutputs := make(map[int64]float64, len(layer))
		for _, node := range layer {
			nodeOutput := node.Activate(layerInputs[node.ID()], layerWeights[node.ID()])
			node.SetActivation(nodeOutput)
			nodeOutputs[node.ID()] = nodeOutput
		}

		// If we're on the output layer, then don't need to calculate inputs for next nodes...
		if layerI == len(n.layers)-1 {
			for i, node := range layer {
				output[i] = nodeOutputs[node.ID()]
			}
			return output, nil
		}

		layerInputs = make(map[int64][]float64)
		layerWeights = make(map[int64][]float64)

		// Rebuild inputs based on the links from nodes in this layer
		for _, conn := range n.connections[layerI] {
			if !conn.Enabled() {
				continue
			}
			if _, ok := layerInputs[conn.To()]; !ok {
				layerInputs[conn.To()] = make([]float64, 0)
			}
			if _, ok := layerWeights[conn.To()]; !ok {
				layerWeights[conn.To()] = make([]float64, 0)
			}
			layerInputs[conn.To()] = append(layerInputs[conn.To()], nodeOutputs[conn.From()])
			layerWeights[conn.To()] = append(layerWeights[conn.To()], conn.weight)
		}
	}

	return output, nil
}

func (n *FeedForward) Copy() NeuralNetwork {
	layersCp := make([]Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		layersCp[i] = make(Layer, len(n.layers[i]))
		for ni := 0; ni < len(n.layers[i]); ni++ {
			layersCp[i][ni] = n.layers[i][ni].Copy()
		}
	}

	connectionsCp := make([]LayerConnections, len(n.connections))
	for i := 0; i < len(n.connections); i++ {
		connectionsCp[i] = make(LayerConnections, len(n.connections[i]))
		for ni := 0; ni < len(n.connections[i]); ni++ {
			connectionsCp[i][ni] = n.connections[i][ni].Copy()
		}
	}
	return NewFeedForward(n.id, layersCp, connectionsCp)
}
