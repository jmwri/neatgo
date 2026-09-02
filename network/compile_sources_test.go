package network_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A connection into an input or bias node is never evaluated, so it must not
// count as a connection and must not be able to form a cycle: a network that
// would run fine has no business being rejected for a loop through an edge
// that does nothing.
func TestCompile_IgnoresConnectionsIntoSources(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: network.Input},
		{ID: 2, Type: network.Bias},
		{ID: 3, Type: network.Output, ActivationFn: network.Identity},
	}
	connections := []network.Connection{
		{ID: 1, From: 1, To: 3, Weight: 2, Enabled: true},
		{ID: 2, From: 2, To: 3, Weight: .5, Enabled: true},
		// Both point back into sources. Together with the edges above they
		// would look like cycles if they were counted.
		{ID: 3, From: 3, To: 1, Weight: 9, Enabled: true},
		{ID: 4, From: 3, To: 2, Weight: 9, Enabled: true},
	}

	net, err := network.Compile(nodes, connections)
	require.NoError(t, err)
	assert.Equal(t, 2, net.NumConnections())

	output, err := net.Activate([]float64{3})
	require.NoError(t, err)
	assert.Equal(t, []float64{6.5}, output)
}
