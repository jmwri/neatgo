package network

import (
	"fmt"
	"sync"
)

// Activate the network defined by nodes and connections with the given input.
// The calculations in each node and connection are executed in their own goroutines.
func Activate(nodes []*Node, connections []*Connection, input []float64) ([]float64, error) {
	// nodeActivation contains channels for passing input/output from a node
	type nodeActivation struct {
		node *Node
		in   []chan float64
		out  []chan float64
	}
	nodeActivations := make([]nodeActivation, len(nodes))
	// nodeActivationIndex provides a lookup map from nodeID > nodeActivations index
	nodeActivationIndex := make(map[int]int)

	// connectionActivation contains channels for passing input/output from a connection
	type connectionActivation struct {
		connection *Connection
		in         chan float64
		out        chan float64
	}
	connectionActivations := make([]connectionActivation, len(connections))

	inputChans := make([]chan float64, 0)
	outputChans := make([]chan float64, 0)

	// Create a nodeActivation for each node
	for i, node := range nodes {
		nodeActivations[i] = nodeActivation{
			node: node,
			in:   make([]chan float64, 0),
			out:  make([]chan float64, 0),
		}
		// If the node is input, then create an input channel.
		if node.Type == Input {
			inputChan := make(chan float64)
			inputChans = append(inputChans, inputChan)
			nodeActivations[i].in = append(nodeActivations[i].in, inputChan)
		}
		// If the node is output, then create an output channel.
		if node.Type == Output {
			outputChan := make(chan float64)
			outputChans = append(outputChans, outputChan)
			nodeActivations[i].out = append(nodeActivations[i].out, outputChan)
		}
		// Add to index to provide easy lookup later
		nodeActivationIndex[node.ID] = i
	}

	// Create a connectionActivation for each connection
	for i, connection := range connections {
		inCh := make(chan float64)
		outCh := make(chan float64)
		connectionActivations[i] = connectionActivation{
			connection: connection,
			in:         inCh,
			out:        outCh,
		}

		// Add an in/out channel to the connected nodes, and store within connection.
		fromNodeActivationIndex := nodeActivationIndex[connection.From]
		toNodeActivationIndex := nodeActivationIndex[connection.To]

		nodeActivations[fromNodeActivationIndex].out = append(nodeActivations[fromNodeActivationIndex].out, inCh)
		nodeActivations[toNodeActivationIndex].in = append(nodeActivations[toNodeActivationIndex].in, outCh)
	}

	output := make([]float64, len(outputChans))
	// Assert that the provided input matches the number of input nodes
	if len(input) != len(inputChans) {
		return output, fmt.Errorf("input does not match network")
	}

	wg := sync.WaitGroup{}
	wg.Add(len(nodeActivations))
	// Start a goroutine for each node
	for _, activation := range nodeActivations {
		go func(activation nodeActivation) {
			defer wg.Done()
			// Sum the inputs to the node, add the bias, and run the activation function.
			state := activation.node.Bias
			for _, inputCh := range activation.in {
				state += <-inputCh
			}
			activated := activation.node.ActivationFn(state)
			// Send the activated value to all connected nodes
			for _, outputCh := range activation.out {
				outputCh <- activated
				close(outputCh)
			}
		}(activation)
	}

	wg.Add(len(connectionActivations))
	// Start a goroutine for each connection
	for _, activation := range connectionActivations {
		go func(activation connectionActivation) {
			defer wg.Done()
			// Multiply the inbound value by the connections weight
			inValue := <-activation.in
			activated := inValue * activation.connection.Weight
			// Send the activated value to the connected node
			activation.out <- activated
			close(activation.out)
		}(activation)
	}

	wg.Add(len(input))
	// Start a goroutine for passing input into the network
	for i, value := range input {
		go func(inputChan chan float64, value float64) {
			defer wg.Done()
			// Send each input value to the input node
			inputChan <- value
			close(inputChan)
		}(inputChans[i], value)
	}

	wg.Add(len(outputChans))
	// Start a goroutine for receiving output values from the network
	for i, outputChan := range outputChans {
		go func(outputChan chan float64, i int, output []float64) {
			defer wg.Done()
			// Receive the output value, and update the output slice
			output[i] = <-outputChan
		}(outputChan, i, output)
	}

	// Wait for all routines to finish
	wg.Wait()

	// Return the calculated output
	return output, nil
}
