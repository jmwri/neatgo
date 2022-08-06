package network_test

import (
	"github.com/jmwri/neatgo/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestActivate(t *testing.T) {
	noActivation := func(x float64) float64 {
		return x
	}
	nodes := []*network.Node{
		{
			ID:           1,
			Type:         network.Input,
			Bias:         0,
			ActivationFn: noActivation,
		},
		{
			ID:           2,
			Type:         network.Input,
			Bias:         0,
			ActivationFn: noActivation,
		},
		{
			ID:           3,
			Type:         network.Hidden,
			Bias:         0,
			ActivationFn: noActivation,
		},
		{
			ID:           4,
			Type:         network.Output,
			Bias:         1,
			ActivationFn: noActivation,
		},
	}
	connections := []*network.Connection{
		{
			ID:     6,
			From:   1,
			To:     3,
			Weight: .8,
		},
		{
			ID:     7,
			From:   2,
			To:     3,
			Weight: .5,
		},
		{
			ID:     8,
			From:   3,
			To:     4,
			Weight: 1,
		},
	}
	/**
	1 - *.8 > 0.8 \
	               1.8 - *1 > 1.8 + 1 = 2.8
	2 - *.5 > 1   /
	*/
	expected := []float64{2.8}
	output, err := network.Activate(nodes, connections, []float64{1.0, 2.0})
	assert.NoErrorf(t, err, "unexpected error from Activate")
	assert.Equal(t, expected, output)
}
