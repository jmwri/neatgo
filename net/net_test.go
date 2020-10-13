package net_test

import (
	"neatgo/net"
	"neatgo/util"
	"testing"
)

func TestFeedForward_Activate(t *testing.T) {
	var n net.NeuralNetwork
	layerDefinitions := []net.LayerDefinition{
		net.NewLayerDefinition(2, 0, 0, nil, nil),
		net.NewLayerDefinition(3, 0, 0, nil, nil),
		net.NewLayerDefinition(2, 0, 0, nil, nil),
	}
	n, _ = net.NewFeedForwardFromDefinition(1, layerDefinitions)

	tests := []struct {
		input    []float64
		expected []float64
	}{
		{
			input:    []float64{1, 1},
			expected: []float64{6, 6},
		},
		{
			input:    []float64{0, 1},
			expected: []float64{3, 3},
		},
	}

	for _, test := range tests {
		outputs, err := n.Activate(test.input)
		if err != nil {
			t.Fatalf("activation failed: %s", err)
		}
		if !util.SliceOfFloatEqual(outputs, test.expected) {
			t.Errorf("expected output %v, got %v", test.expected, outputs)
		}
	}
}
