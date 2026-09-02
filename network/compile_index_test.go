package network_test

import (
	"errors"
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Node IDs are looked up through a dense table when they are packed and a map
// when they are spread out. Both paths must compile the same network and
// both must catch a duplicate.
func TestCompile_HandlesPackedAndSparseNodeIDs(t *testing.T) {
	for name, ids := range map[string][3]int{
		"packed":   {1, 2, 3},
		"sparse":   {7, 100000, 9000000},
		"negative": {-50, 0, 12},
	} {
		t.Run(name, func(t *testing.T) {
			nodes := []network.Node{
				{ID: ids[0], Type: network.Input},
				{ID: ids[1], Type: network.Hidden, Bias: .5, ActivationFn: network.Identity},
				{ID: ids[2], Type: network.Output, ActivationFn: network.Identity},
			}
			connections := []network.Connection{
				{ID: 1, From: ids[0], To: ids[1], Weight: 2, Enabled: true},
				{ID: 2, From: ids[1], To: ids[2], Weight: 3, Enabled: true},
				// Refers to a node that does not exist, and is dropped.
				{ID: 3, From: 424242, To: ids[2], Weight: 3, Enabled: true},
			}
			net, err := network.Compile(nodes, connections)
			require.NoError(t, err)
			assert.Equal(t, 2, net.NumConnections())

			output, err := net.Activate([]float64{1})
			require.NoError(t, err)
			// (1*2 + .5) * 3
			assert.InDelta(t, 7.5, output[0], 1e-12)

			nodes = append(nodes, network.Node{ID: ids[1], Type: network.Hidden, ActivationFn: network.Identity})
			_, err = network.Compile(nodes, connections)
			assert.True(t, errors.Is(err, network.ErrDuplicateNode), "want ErrDuplicateNode, got %v", err)
		})
	}
}

func TestActivationFunctions_ClampAndPowers(t *testing.T) {
	assert.Equal(t, 9.0, network.SquareFn(-3))
	assert.Equal(t, -27.0, network.CubeFn(-3))
	assert.Equal(t, 1.0, network.ClampedFn(5))
	assert.Equal(t, -1.0, network.ClampedFn(-5))
	assert.Equal(t, .25, network.ClampedFn(.25))
	assert.Equal(t, math.Log(1e-7), network.LogFn(-1))
	assert.InDelta(t, 1, network.SigmoidFn(1000), 1e-12)
	assert.InDelta(t, 0, network.SigmoidFn(-1000), 1e-12)
	assert.Equal(t, math.Exp(60), network.ExpFn(1e6))
	// NaN is passed through rather than clamped to a bound, the same as
	// math.Max and math.Min would have done.
	assert.True(t, math.IsNaN(network.SigmoidFn(math.NaN())))
	assert.True(t, math.IsNaN(network.ClampedFn(math.NaN())))
}
