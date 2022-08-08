package network

import (
	"fmt"
	"sync"
)

// Activate the network defined by nodes and connections with the given input.
// The calculations in each node and connection are executed in their own goroutines.
func Activate(nodes []Node, connections []Connection, input []float64) ([]float64, error) {
	// nodeActivation contains channels for passing input/output from a node
	type nodeActivation struct {
		node Node
		in   []chan float64
		out  []chan float64
	}
	nodeActivations := make([]nodeActivation, len(nodes))
	// nodeActivationIndex provides a lookup map from nodeID > nodeActivations index
	nodeActivationIndex := make(map[int]int)

	// connectionActivation contains channels for passing input/output from a connection
	type connectionActivation struct {
		connection Connection
		in         chan float64
		out        chan float64
	}
	connectionActivations := make([]connectionActivation, len(connections))

	inputChans := make([]chan float64, 0)
	biasChans := make([]chan float64, 0)
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
		// If the node is bias, then create a bias channel.
		if node.Type == Bias {
			biasChan := make(chan float64)
			biasChans = append(biasChans, biasChan)
			nodeActivations[i].in = append(nodeActivations[i].in, biasChan)
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

	nodesActivatedWg := sync.WaitGroup{}
	nodesActivatedWg.Add(len(nodeActivations))
	// Start a goroutine for each node
	for _, activation := range nodeActivations {
		go func(activation nodeActivation) {
			defer nodesActivatedWg.Done()
			// Sum the inputs to the node, add the bias, and run the activation function.
			state := activation.node.Bias
			subInputCh := make(chan float64)

			// Sum state ready for activation
			sumWg := sync.WaitGroup{}
			sumWg.Add(1)
			go func() {
				defer sumWg.Done()
				for inputValue := range subInputCh {
					state += inputValue
				}
			}()

			funnelInputsWg := sync.WaitGroup{}
			funnelInputsWg.Add(len(activation.in))
			for _, inputCh := range activation.in {
				// Read each input in separate gorountines so that we don't get deadlocks
				go func(inputCh <-chan float64, subInputCh chan<- float64) {
					defer funnelInputsWg.Done()
					inputValue, ok := <-inputCh
					if ok {
						subInputCh <- inputValue
					}
				}(inputCh, subInputCh)
			}

			// Wait for all inputs to be funneled
			funnelInputsWg.Wait()
			close(subInputCh)
			// Wait for the summed state
			sumWg.Wait()

			// Get the node activation value
			activated := activation.node.ActivationFn(state)

			// Send the activated value to all connected nodes
			outputFanOutWg := sync.WaitGroup{}
			outputFanOutWg.Add(len(activation.out))
			for _, outputCh := range activation.out {
				go func(outputCh chan<- float64) {
					defer outputFanOutWg.Done()
					outputCh <- activated
					close(outputCh)
				}(outputCh)
			}
		}(activation)
	}

	// Start a goroutine for each connection
	connectionsActivatedWg := sync.WaitGroup{}
	connectionsActivatedWg.Add(len(connectionActivations))
	for _, activation := range connectionActivations {
		go func(activation connectionActivation) {
			defer connectionsActivatedWg.Done()
			outputSentWg := sync.WaitGroup{}
			// Multiply the inbound value by the connections weight
			inValue, ok := <-activation.in
			if ok && activation.connection.Enabled {
				activated := inValue * activation.connection.Weight
				// Send the activated value to the connected node
				outputSentWg.Add(1)
				go func() {
					defer outputSentWg.Done()
					activation.out <- activated
				}()
			} else {
				close(activation.out)
			}
			outputSentWg.Wait()
		}(activation)
	}

	inputSentWg := sync.WaitGroup{}
	inputSentWg.Add(len(input))
	// Start a goroutine for passing input into the network
	for i, value := range input {
		go func(inputChan chan float64, value float64) {
			defer inputSentWg.Done()
			// Send each input value to the input node
			inputChan <- value
			close(inputChan)
		}(inputChans[i], value)
	}

	biasSentWg := sync.WaitGroup{}
	biasSentWg.Add(len(biasChans))
	// Start a goroutine for passing input into the network
	for _, biasChan := range biasChans {
		go func(biasChan chan float64) {
			defer biasSentWg.Done()
			// Send each input value to the input node
			biasChan <- 1.0
			close(biasChan)
		}(biasChan)
	}

	outputSentWg := sync.WaitGroup{}
	outputSentWg.Add(len(outputChans))
	// Start a goroutine for receiving output values from the network
	for i, outputChan := range outputChans {
		go func(outputChan chan float64, i int, output []float64) {
			defer outputSentWg.Done()
			// Receive the output value, and update the output slice
			output[i] = <-outputChan
		}(outputChan, i, output)
	}

	inputSentWg.Wait()
	biasSentWg.Wait()
	nodesActivatedWg.Wait()
	connectionsActivatedWg.Wait()
	outputSentWg.Wait()

	// Return the calculated output
	return output, nil
}
