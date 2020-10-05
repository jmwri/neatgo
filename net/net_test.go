package net_test

import (
	"neatgo/activation"
	"neatgo/net"
	"neatgo/util"
	"testing"
)

func TestFeedForward_Activate(t *testing.T) {
	var n net.NeuralNetwork
	layerDefinitions := []net.LayerDefinition{
		{
			NumNodes:     2,
			ActivationFn: activation.Nil,
		},
		{
			NumNodes:     3,
			ActivationFn: activation.Nil,
		},
		{
			NumNodes:     2,
			ActivationFn: activation.Nil,
		},
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
