package network_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
)

// Input nodes are sensors: whatever the caller passes in must arrive at the
// rest of the network untouched, with no bias added and no activation applied.
func TestActivate_InputNodesArePassthrough(t *testing.T) {
	nodes := []network.Node{
		// Bias and activation are deliberately hostile here. A correct
		// implementation ignores both on an input node.
		{ID: 1, Type: network.Input, Bias: 7, ActivationFn: network.Sigmoid},
		{ID: 2, Type: network.Output, Bias: 0, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 3, From: 1, To: 2, Weight: 1, Enabled: true},
	}

	output, err := network.Activate(nodes, connections, []float64{2.5})
	assert.NoError(t, err)
	assert.Equal(t, []float64{2.5}, output)
}

func TestActivate_BiasNodeEmitsOne(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Bias, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 3, From: 1, To: 2, Weight: .25, Enabled: true},
	}

	output, err := network.Activate(nodes, connections, nil)
	assert.NoError(t, err)
	assert.Equal(t, []float64{.25}, output)
}

func TestActivate_DisabledConnectionsAreIgnored(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 3, From: 1, To: 2, Weight: 1, Enabled: false},
		{ID: 4, From: 1, To: 2, Weight: 2, Enabled: true},
	}

	output, err := network.Activate(nodes, connections, []float64{3})
	assert.NoError(t, err)
	assert.Equal(t, []float64{6}, output)
}

// A cycle has no valid feed-forward evaluation order. It must be reported, not
// hung on: the previous channel based evaluator would have deadlocked here and
// taken the whole run with it.
func TestActivate_CycleReturnsError(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Hidden, ActivationFn: network.NoActivation},
		{ID: 3, Type: network.Hidden, ActivationFn: network.NoActivation},
		{ID: 4, Type: network.Output, ActivationFn: network.NoActivation},
	}
	connections := []network.Connection{
		{ID: 5, From: 1, To: 2, Weight: 1, Enabled: true},
		{ID: 6, From: 2, To: 3, Weight: 1, Enabled: true},
		{ID: 7, From: 3, To: 2, Weight: 1, Enabled: true},
		{ID: 8, From: 3, To: 4, Weight: 1, Enabled: true},
	}

	_, err := network.Activate(nodes, connections, []float64{1})
	assert.ErrorContains(t, err, "cycle")
}

func TestActivate_InputLengthMismatch(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.NoActivation},
	}

	_, err := network.Activate(nodes, nil, []float64{1, 2})
	assert.ErrorContains(t, err, "expects 1 inputs")
}

func TestActivate_UnknownActivationReturnsError(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: "not-registered"},
	}

	_, err := network.Activate(nodes, nil, []float64{1})
	assert.ErrorContains(t, err, "unknown activation function")
}

// Overriding a registered activation must replace it, not list its name twice
// and skew every random activation choice thereafter.
func TestActivationRegistry_OverrideDoesNotDuplicateName(t *testing.T) {
	before := len(network.ActivationRegistry.Names())
	original := network.ActivationRegistry.Get(network.Sigmoid)
	t.Cleanup(func() { network.ActivationRegistry.Set(network.Sigmoid, original) })

	network.ActivationRegistry.Set(network.Sigmoid, func(x float64) float64 { return x })
	assert.Len(t, network.ActivationRegistry.Names(), before)

	count := 0
	for _, name := range network.ActivationRegistry.Names() {
		if name == network.Sigmoid {
			count++
		}
	}
	assert.Equal(t, 1, count)
}

func TestGaussFn_PeaksAtZeroAndDecays(t *testing.T) {
	// exp(-5x^2): 1 at the origin, decaying away from it. The original wrote
	// exp((-5x)^2), which explodes instead of decaying.
	assert.InDelta(t, 1, network.GaussFn(0), 1e-12)
	assert.Less(t, network.GaussFn(1), network.GaussFn(0))
	assert.Less(t, network.GaussFn(3), network.GaussFn(1))
	assert.InDelta(t, network.GaussFn(-1), network.GaussFn(1), 1e-12)
	assert.LessOrEqual(t, network.GaussFn(100), 1.0)
}
