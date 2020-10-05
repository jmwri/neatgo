package neat

import (
	"neatgo/activation"
	"neatgo/net"
	"neatgo/util"
)

type Genome interface {
	net.NeuralNetwork
}

func NewFeedForwardGenome(id int64, numBias int, numInput int, numHidden int, numOutput int) *FeedForwardGenome {
	var nodeID int64 = 1
	biasNodes := make([]net.BiasNode, numBias)
	for i := 0; i < numBias; i++ {
		biasNodes = append(biasNodes, net.NewBiasNode(nodeID, util.RandFloat(0, 1)))
		nodeID++
	}
	inputNodes := make([]net.InputNode, numInput)
	for i := 0; i < numInput; i++ {
		inputNodes = append(inputNodes, net.NewInputNode(nodeID))
		nodeID++
	}
	hiddenNodes := make([]net.HiddenNode, numHidden)
	for i := 0; i < numHidden; i++ {
		hiddenNodes = append(hiddenNodes, net.NewHiddenNode(nodeID, activation.RandFn()))
		nodeID++
	}
	outputNodes := make([]net.OutputNode, numOutput)
	for i := 0; i < numOutput; i++ {
		outputNodes = append(outputNodes, net.NewOutputNode(nodeID, activation.RandFn()))
		nodeID++
	}

	connDefinitions := make([]*net.ConnectionDefinition, 0)
	for _, biasNode := range biasNodes {
		for _, hiddenNode := range hiddenNodes {
			connDefinitions = append(connDefinitions, net.NewConnectionDefinition(biasNode, hiddenNode, util.RandFloat(-1, 1)))
		}
	}

	for _, inputNode := range inputNodes {
		for _, hiddenNode := range hiddenNodes {
			connDefinitions = append(connDefinitions, net.NewConnectionDefinition(inputNode, hiddenNode, util.RandFloat(-1, 1)))
		}
	}

	for _, hiddenNode := range hiddenNodes {
		for _, outputNode := range outputNodes {
			connDefinitions = append(connDefinitions, net.NewConnectionDefinition(hiddenNode, outputNode, util.RandFloat(-1, 1)))
		}
	}

	net.AddConnections(connDefinitions)

	n := net.NewFeedForward(id, biasNodes, inputNodes, hiddenNodes, outputNodes)
	return &FeedForwardGenome{
		FeedForward: n,
	}
}

type FeedForwardGenome struct {
	*net.FeedForward
}
