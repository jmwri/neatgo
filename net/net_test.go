package net_test

import (
	"neatgo/activation"
	"neatgo/net"
	"neatgo/util"
	"testing"
)

func TestFeedForward_Activate(t *testing.T) {
	bias := net.NewBiasNode(1, 1)
	biasNodes := []net.BiasNode{
		bias,
	}

	inputA := net.NewInputNode(1)
	inputNodes := []net.InputNode{
		inputA,
	}

	hiddenA := net.NewHiddenNode(2, activation.Nil)
	hiddenB := net.NewHiddenNode(3, activation.Nil)
	hiddenNodes := []net.HiddenNode{
		hiddenA,
	}

	outputA := net.NewOutputNode(4, activation.Nil)
	outputB := net.NewOutputNode(5, activation.Nil)
	outputNodes := []net.OutputNode{
		outputA,
		outputB,
	}

	definitions := []*net.ConnectionDefinition{
		net.NewConnectionDefinition(bias, hiddenA, 1),
		net.NewConnectionDefinition(bias, hiddenB, 1),
		net.NewConnectionDefinition(inputA, hiddenA, 1),
		net.NewConnectionDefinition(inputA, hiddenB, 1),
		net.NewConnectionDefinition(hiddenA, outputA, 1),
		net.NewConnectionDefinition(hiddenA, outputB, 1),
		net.NewConnectionDefinition(hiddenB, outputA, 1),
		net.NewConnectionDefinition(hiddenB, outputB, 1),
	}

	net.AddConnections(definitions)

	var n net.NeuralNetwork = net.NewFeedForward(1, biasNodes, inputNodes, hiddenNodes, outputNodes)

	tests := []struct {
		input    []float64
		expected []float64
	}{
		{
			input:    []float64{1},
			expected: []float64{2, 2},
		},
		{
			input:    []float64{0},
			expected: []float64{1, 1},
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
