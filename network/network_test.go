package network_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func compiledTestNetwork(t *testing.T) *network.Network {
	t.Helper()
	nodes, connections, _ := buildNetwork(4, 6, 3)
	net, err := network.Compile(nodes, connections)
	require.NoError(t, err)
	return net
}

// A compiled network is immutable, so one instance must be shareable across
// goroutines. This is what lets a population share a network without the caller
// having to think about it.
func TestNetwork_ActivateIsConcurrencySafe(t *testing.T) {
	net := compiledTestNetwork(t)
	input := []float64{.25, -.5, 1, 0}

	want, err := net.Activate(input)
	require.NoError(t, err)

	const goroutines, iterations = 32, 200
	var wg sync.WaitGroup
	wg.Add(goroutines)
	results := make([][]float64, goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				got, err := net.Activate(input)
				if err != nil {
					t.Error(err)
					return
				}
				results[g] = got
			}
		}(g)
	}
	wg.Wait()

	for g, got := range results {
		assert.Equal(t, want, got, "goroutine %d saw a different result", g)
	}
}

// ActivateInto shares the network but not the output buffer, so concurrent
// callers must each supply their own.
func TestNetwork_ActivateIntoIsConcurrencySafe(t *testing.T) {
	net := compiledTestNetwork(t)
	input := []float64{1, 1, 1, 1}
	want, err := net.Activate(input)
	require.NoError(t, err)

	var wg sync.WaitGroup
	wg.Add(16)
	for g := 0; g < 16; g++ {
		go func() {
			defer wg.Done()
			output := make([]float64, net.NumOutputs())
			for i := 0; i < 200; i++ {
				if err := net.ActivateInto(input, output); err != nil {
					t.Error(err)
					return
				}
				assert.Equal(t, want, output)
			}
		}()
	}
	wg.Wait()
}

func TestNetwork_Shape(t *testing.T) {
	nodes, connections, _ := buildNetwork(3, 5, 2)
	net, err := network.Compile(nodes, connections)
	require.NoError(t, err)

	assert.Equal(t, 3, net.NumInputs())
	assert.Equal(t, 2, net.NumOutputs())
	assert.Equal(t, 10, net.NumNodes())
	assert.Equal(t, 25, net.NumConnections())
}

// Disabled connections are dropped at compile time, so a complexity penalty
// built on NumConnections counts only what actually runs.
func TestNetwork_NumConnectionsCountsOnlyEnabled(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 3, From: 1, To: 2, Weight: 1, Enabled: true},
		{ID: 4, From: 1, To: 2, Weight: 1, Enabled: false},
	}

	net, err := network.Compile(nodes, connections)
	require.NoError(t, err)
	assert.Equal(t, 1, net.NumConnections())
}

func TestCompile_RejectsDuplicateNodeIDs(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 1, Type: network.Output, ActivationFn: network.NoActivation},
	}
	_, err := network.Compile(nodes, nil)
	assert.ErrorContains(t, err, "duplicate node id")
}

func TestNetwork_ActivateIntoRejectsWrongSizedBuffer(t *testing.T) {
	net := compiledTestNetwork(t)
	err := net.ActivateInto(make([]float64, net.NumInputs()), make([]float64, net.NumOutputs()+1))
	assert.ErrorContains(t, err, "buffer")
}

// Output order follows the order the nodes were given in, not the order the
// network happens to evaluate them in.
func TestNetwork_OutputOrderFollowsNodeOrder(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.NoActivation},
		{ID: 3, Type: network.Output, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 4, From: 1, To: 2, Weight: 10, Enabled: true},
		{ID: 5, From: 1, To: 3, Weight: 100, Enabled: true},
	}

	net, err := network.Compile(nodes, connections)
	require.NoError(t, err)
	output, err := net.Activate([]float64{1})
	require.NoError(t, err)
	assert.Equal(t, []float64{10, 100}, output)
}

func BenchmarkNetworkActivate(b *testing.B) {
	nodes, connections, input := buildNetwork(8, 16, 16, 4)

	b.Run("Compile+Activate (convenience)", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := network.Activate(nodes, connections, input); err != nil {
				b.Fatal(err)
			}
		}
	})

	net, err := network.Compile(nodes, connections)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("compiled Activate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := net.Activate(input); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("compiled ActivateInto", func(b *testing.B) {
		b.ReportAllocs()
		output := make([]float64, net.NumOutputs())
		for i := 0; i < b.N; i++ {
			if err := net.ActivateInto(input, output); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("compiled ActivateInto parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			output := make([]float64, net.NumOutputs())
			for pb.Next() {
				if err := net.ActivateInto(input, output); err != nil {
					b.Fatal(err)
				}
			}
		})
	})
}

func BenchmarkCompile(b *testing.B) {
	for _, layerSizes := range [][]int{{2, 1}, {2, 3, 1}, {8, 16, 16, 4}} {
		nodes, connections, _ := buildNetwork(layerSizes...)
		b.Run(fmt.Sprintf("%v", layerSizes), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := network.Compile(nodes, connections); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
