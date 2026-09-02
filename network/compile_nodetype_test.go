package network_test

import (
	"errors"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
)

// A misspelt node type in a hand-written genome must be an error, not a
// hidden node that quietly shifts which input feeds what.
func TestCompile_RejectsUnknownNodeType(t *testing.T) {
	nodes := []network.Node{
		{ID: 1, Type: "inptu", ActivationFn: network.NoActivation},
		{ID: 2, Type: network.Output, ActivationFn: network.Identity},
	}
	_, err := network.Compile(nodes, nil)
	assert.True(t, errors.Is(err, network.ErrUnknownNodeType), "want ErrUnknownNodeType, got %v", err)
	assert.ErrorContains(t, err, `"inptu"`)
}
