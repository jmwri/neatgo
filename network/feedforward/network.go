package feedforward

import (
	"context"
	"fmt"
	"github.com/jmwri/neatgo/util"
	"io"
	"sync"
	"time"
)

func NewNetwork(input, hidden, output []Node, connections []Connection) *Network {
	return &Network{
		input:       input,
		hidden:      hidden,
		output:      output,
		connections: connections,
		stats:       NetworkStats{},
	}
}

type Network struct {
	input, hidden, output []Node
	connections           []Connection
	stats                 NetworkStats
}

func newNetworkStats(started, finished time.Time) NetworkStats {
	return NetworkStats{
		StartedAt:  started,
		FinishedAt: finished,
		Duration:   finished.Sub(started),
	}
}

type NetworkStats struct {
	StartedAt  time.Time
	FinishedAt time.Time
	Duration   time.Duration
}

func (s NetworkStats) Printf(writer io.Writer) (n int, err error) {
	return fmt.Fprintf(
		writer,
		"Ran from '%s' to '%s'\nDuration: %s\n",
		s.StartedAt.Format(time.RFC3339Nano),
		s.FinishedAt.Format(time.RFC3339Nano),
		s.Duration,
	)
}

func (n *Network) Activate(ctx context.Context, in ...float64) error {
	if len(in) != len(n.input) {
		return fmt.Errorf("number of inputs %d does not match number of input nodes %d", len(in), len(n.input))
	}

	startedAt := time.Now()

	// For node  processing
	wg := &sync.WaitGroup{}
	// Channels to receive the activated nodes and connections
	activatedNodeCh := make(chan activatedNodeEvent)
	activatedConnCh := make(chan activatedConnEvent)

	// Once all processing is complete:
	// - finalise and update network stats
	// - close activation event channels
	go func() {
		wg.Wait()
		finishedAt := time.Now()
		n.stats = newNetworkStats(startedAt, finishedAt)
		close(activatedNodeCh)
		close(activatedConnCh)
	}()

	// Start workers to process connections
	connInChans := make(map[int64]chan float64)
	connOutChans := make(map[int64]chan float64)
	for _, conn := range n.connections {
		if !conn.Enabled() {
			continue
		}
		connInChans[conn.ID()] = make(chan float64)
		connOutChans[conn.ID()] = make(chan float64)
		go func(conn Connection) {
			inCh := connInChans[conn.ID()]
			outCh := connOutChans[conn.ID()]
			defer close(outCh)
			inputs := util.ReadAllChan(inCh)
			conn = conn.Activate(inputs...)
			outCh <- conn.Value()
			activatedConnCh <- activatedConnEvent{c: conn}
		}(conn)
	}

	// Start workers to process input nodes
	wg.Add(len(n.input))
	for i, node := range n.input {
		nodeOutputChans := make([]chan<- float64, 0)
		node = node.Activate(in[i])
		for _, conn := range n.connections {
			if !conn.Enabled() {
				continue
			}
			if conn.From() == node.ID() {
				nodeOutputChans = append(nodeOutputChans, connInChans[conn.ID()])
			}
		}

		go func(inputIdx int, node Node) {
			defer wg.Done()
			node = node.Activate(in[inputIdx])
			for _, outCh := range nodeOutputChans {
				go func(nodeIdx int, node Node, outCh chan<- float64) {
					defer close(outCh)
					outCh <- node.Value()
					activatedNodeCh <- activatedNodeEvent{
						t: input,
						i: nodeIdx,
						n: node,
					}
				}(inputIdx, node, outCh)
			}
		}(i, node)
	}

	// Start workers to process hidden nodes
	wg.Add(len(n.hidden))
	for i, node := range n.hidden {
		nodeInputChans := make([]<-chan float64, 0)
		nodeOutputChans := make([]chan<- float64, 0)
		for _, conn := range n.connections {
			if !conn.Enabled() {
				continue
			}
			if conn.To() == node.ID() {
				nodeInputChans = append(nodeInputChans, connOutChans[conn.ID()])
			}
			if conn.From() == node.ID() {
				nodeOutputChans = append(nodeOutputChans, connInChans[conn.ID()])
			}
		}

		go func(nodeIdx int, node Node) {
			defer wg.Done()
			nodeInputCh := util.AggregateChannels(nodeInputChans)
			inputs := util.ReadAllChan(nodeInputCh)
			node = node.Activate(inputs...)
			for _, outCh := range nodeOutputChans {
				go func(nodeIdx int, node Node, outCh chan<- float64) {
					defer close(outCh)
					outCh <- node.Value()
					activatedNodeCh <- activatedNodeEvent{
						t: hidden,
						i: nodeIdx,
						n: node,
					}
				}(nodeIdx, node, outCh)
			}
		}(i, node)
	}

	// Start workers to process output nodes
	wg.Add(len(n.output))
	for i, node := range n.output {
		nodeInputChans := make([]<-chan float64, 0)
		for _, conn := range n.connections {
			if !conn.Enabled() {
				continue
			}
			if conn.To() == node.ID() {
				nodeInputChans = append(nodeInputChans, connOutChans[conn.ID()])
			}
		}

		go func(nodeIdx int, node Node) {
			defer wg.Done()
			nodeInputCh := util.AggregateChannels(nodeInputChans)
			inputs := util.ReadAllChan(nodeInputCh)
			node = node.Activate(inputs...)
			activatedNodeCh <- activatedNodeEvent{
				t: output,
				i: nodeIdx,
				n: node,
			}
		}(i, node)
	}

	// Start workers that will update the network nodes + connections upon activation
	updateNetworkWg := &sync.WaitGroup{}
	updateNetworkWg.Add(2)
	updateNetworkMu := &sync.Mutex{}
	go func() {
		defer updateNetworkWg.Done()
		for event := range activatedNodeCh {
			updateNetworkMu.Lock()
			switch event.t {
			case input:
				n.input[event.i] = event.n
			case hidden:
				n.hidden[event.i] = event.n
			case output:
				n.output[event.i] = event.n
			}
			updateNetworkMu.Unlock()
		}
	}()
	go func() {
		defer updateNetworkWg.Done()
		for event := range activatedConnCh {
			updateNetworkMu.Lock()
			n.connections[event.i] = event.c
			updateNetworkMu.Unlock()
		}
	}()

	// Wait for network to update
	updateNetworkWg.Wait()

	return nil
}

func (n *Network) Output() []float64 {
	outputValues := make([]float64, len(n.output))
	for i, node := range n.output {
		outputValues[i] = node.Value()
	}
	return outputValues
}

func (n *Network) Stats() NetworkStats {
	return n.stats
}

type activatedNodeEvent struct {
	t nodeType
	i int
	n Node
}
type activatedConnEvent struct {
	i int
	c Connection
}
