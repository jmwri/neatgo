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
	nodes := []network.Node{
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
			Type:         network.Bias,
			Bias:         0,
			ActivationFn: noActivation,
		},
		{
			ID:           4,
			Type:         network.Hidden,
			Bias:         0,
			ActivationFn: noActivation,
		},
		{
			ID:           5,
			Type:         network.Output,
			Bias:         1,
			ActivationFn: noActivation,
		},
	}
	connections := []network.Connection{
		{
			ID:      6,
			From:    1,
			To:      4,
			Weight:  .8,
			Enabled: true,
		},
		{
			ID:      7,
			From:    2,
			To:      4,
			Weight:  .5,
			Enabled: true,
		},
		{
			ID:      8,
			From:    4,
			To:      5,
			Weight:  1,
			Enabled: true,
		},
		{
			ID:      9,
			From:    3,
			To:      4,
			Weight:  .5,
			Enabled: true,
		},
		{
			ID:      10,
			From:    3,
			To:      5,
			Weight:  .5,
			Enabled: true,
		},
	}
	/**
	1 - *.5 > .5 -------------------\
	             \                   \
	1 - *.8 > 0.8 \                   \
	               2.3 - *1 > 2.3 + 1 +.5 = 3.8
	2 - *.5 > 1   /
	*/
	expected := []float64{3.8}
	output, err := network.Activate(nodes, connections, []float64{1.0, 2.0})
	assert.NoErrorf(t, err, "unexpected error from Activate")
	assert.Equal(t, expected, output)
}
