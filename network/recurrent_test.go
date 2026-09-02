package network_test

import (
	"errors"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loopingNodes is one input feeding one output, with the output also feeding
// itself. Identity activations everywhere, so what comes out is arithmetic that
// can be checked by hand rather than a number from a sigmoid.
func loopingNodes() ([]network.Node, []network.Connection) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input},
		{ID: 2, Type: network.Output, ActivationFn: network.Identity},
	}
	connections := []network.Connection{
		{ID: 1, From: 1, To: 2, Weight: 1, Enabled: true},
		// The loop: the output reads what it was last time, halved.
		{ID: 2, From: 2, To: 2, Weight: 0.5, Enabled: true},
	}
	return nodes, connections
}

func TestCompileRejectsALoopAndCompileRecurrentDoesNot(t *testing.T) {
	nodes, connections := loopingNodes()

	_, err := network.Compile(nodes, connections)
	require.Error(t, err)
	assert.True(t, errors.Is(err, network.ErrCycle), "want ErrCycle, got %v", err)

	net, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)
	assert.True(t, net.IsRecurrent())
}

// TestRecurrentNetworkRemembers is the whole point of the thing: the same input
// twice must not give the same answer twice, because the second time the network
// has something to remember.
func TestRecurrentNetworkRemembers(t *testing.T) {
	nodes, connections := loopingNodes()
	net, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)

	mem := net.NewMemory()
	output := make([]float64, net.NumOutputs())

	// First step: nothing remembered, so just the input through a weight of 1.
	require.NoError(t, net.Step(mem, []float64{1}, output))
	assert.InDelta(t, 1.0, output[0], 1e-9)

	// Second: the same input, plus half of what the output was last time.
	require.NoError(t, net.Step(mem, []float64{1}, output))
	assert.InDelta(t, 1.5, output[0], 1e-9)

	// Third: 1 + 1.5/2.
	require.NoError(t, net.Step(mem, []float64{1}, output))
	assert.InDelta(t, 1.75, output[0], 1e-9)

	// Forgetting puts it back to where it started.
	mem.Reset()
	require.NoError(t, net.Step(mem, []float64{1}, output))
	assert.InDelta(t, 1.0, output[0], 1e-9)
}

func TestRecurrentNetworkRefusesToRunWithoutMemory(t *testing.T) {
	nodes, connections := loopingNodes()
	net, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)

	output := make([]float64, net.NumOutputs())
	assert.True(t, errors.Is(net.ActivateInto([]float64{1}, output), network.ErrNeedsMemory))
	assert.True(t, errors.Is(net.Step(nil, []float64{1}, output), network.ErrNeedsMemory))
}

func TestMemoryFromAnotherNetworkIsRejected(t *testing.T) {
	nodes, connections := loopingNodes()
	net, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)

	other := compiledTestNetwork(t)
	output := make([]float64, net.NumOutputs())
	err = net.Step(other.NewMemory(), []float64{1}, output)
	assert.True(t, errors.Is(err, network.ErrMemorySize), "want ErrMemorySize, got %v", err)
}

// TestRecurrentCompilesFeedForwardTheSameWay keeps the option honest: turning it
// on must not change what a genome without loops in it does.
func TestRecurrentCompilesFeedForwardTheSameWay(t *testing.T) {
	nodes, connections, _ := buildNetwork(4, 6, 3)

	forward, err := network.Compile(nodes, connections)
	require.NoError(t, err)
	looping, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)

	input := []float64{0.3, -0.7, 1, 0.1}
	want := make([]float64, forward.NumOutputs())
	got := make([]float64, looping.NumOutputs())
	require.NoError(t, forward.ActivateInto(input, want))

	mem := looping.NewMemory()
	// Several steps: with no backward connections there is nothing to remember,
	// so the answer must not drift.
	for step := 0; step < 3; step++ {
		require.NoError(t, looping.Step(mem, input, got))
		assert.InDeltaSlice(t, want, got, 1e-9, "step %d", step)
	}
}

// TestRecurrentNetworkIsSafeForConcurrentUse checks the reason Memory belongs to
// the caller: one network, several goroutines, each remembering its own thing.
func TestRecurrentNetworkIsSafeForConcurrentUse(t *testing.T) {
	nodes, connections := loopingNodes()
	net, err := network.CompileRecurrent(nodes, connections)
	require.NoError(t, err)

	const runners = 8
	results := make(chan float64, runners)
	for i := 0; i < runners; i++ {
		go func() {
			mem := net.NewMemory()
			output := make([]float64, net.NumOutputs())
			for step := 0; step < 3; step++ {
				if err := net.Step(mem, []float64{1}, output); err != nil {
					results <- -1
					return
				}
			}
			results <- output[0]
		}()
	}
	for i := 0; i < runners; i++ {
		assert.InDelta(t, 1.75, <-results, 1e-9)
	}
}
