package network

import (
	"errors"
	"fmt"
	"reflect"
)

// ErrNotFeedForward is returned by Program and the batch functions for a
// recurrent network. Its answer depends on what it activated before, so it has
// no meaning as a pure function of one input, which is what a batch computes.
var ErrNotFeedForward = errors.New("network: recurrent network cannot be run as a batch")

// Program is the flat, pointer-free description of a feed-forward network: what
// something that cannot call Activate - a GPU kernel, an exporter - needs in
// order to run it.
//
// Nodes are in evaluation order, so every edge reads a node with a smaller index
// than the one it feeds and a single forward pass computes the network.
type Program struct {
	NumInputs int
	// Outputs is the index in Nodes of each output, in output order.
	Outputs []int
	Nodes   []ProgramNode
	// Edges holds every node's incoming connections back to back; a node owns
	// Edges[FirstEdge:EndEdge].
	Edges []ProgramEdge
}

// ProgramNode is one node of a Program.
type ProgramNode struct {
	Kind ProgramNodeKind
	// InputPos is the position in the input slice, for an input node.
	InputPos int
	Bias     float64
	// Activation names the function a computed node applies to its summed input.
	Activation ActivationFunctionName
	FirstEdge  int
	EndEdge    int
}

// ProgramNodeKind says how a node gets its value.
type ProgramNodeKind uint8

const (
	// ProgramComputed nodes apply their activation to bias plus the weighted sum
	// of their edges.
	ProgramComputed ProgramNodeKind = iota
	// ProgramInput nodes read the input at InputPos.
	ProgramInput
	// ProgramBias nodes are the constant 1.
	ProgramBias
)

// ProgramEdge is one weighted connection into a node.
type ProgramEdge struct {
	// From indexes Nodes.
	From   int
	Weight float64
}

// Program returns the flat form of a feed-forward network. The result shares
// nothing with the network.
func (n *Network) Program() (Program, error) {
	if n.recurrent {
		return Program{}, ErrNotFeedForward
	}
	program := Program{
		NumInputs: n.numInputs,
		Outputs:   append([]int(nil), n.outputs...),
		Nodes:     make([]ProgramNode, len(n.nodes)),
		Edges:     make([]ProgramEdge, 0, n.numConnections),
	}
	for i, node := range n.nodes {
		out := ProgramNode{FirstEdge: len(program.Edges), InputPos: node.inputPos}
		switch node.kind {
		case kindInput:
			out.Kind = ProgramInput
		case kindBias:
			out.Kind = ProgramBias
		default:
			out.Kind = ProgramComputed
			out.Bias = node.bias
			out.Activation = node.activationName
			for _, in := range node.inputs {
				program.Edges = append(program.Edges, ProgramEdge{From: in.from, Weight: in.weight})
			}
		}
		out.EndEdge = len(program.Edges)
		program.Nodes[i] = out
	}
	return program, nil
}

// builtinActivations remembers the functions the registry started with, so that
// IsBuiltinActivation can tell a name that still means what this package says
// from one a caller has since re-registered.
var builtinActivations = func() map[ActivationFunctionName]uintptr {
	fns := make(map[ActivationFunctionName]uintptr)
	for _, name := range ActivationRegistry.Names() {
		fns[name] = reflect.ValueOf(ActivationRegistry.Get(name)).Pointer()
	}
	return fns
}()

// IsBuiltinActivation reports whether name is registered and still refers to the
// function this package registered under it. A GPU implements the built-in
// functions itself, so one that a caller has replaced or added has to run on the
// CPU: the GPU would quietly compute the original instead.
func IsBuiltinActivation(name ActivationFunctionName) bool {
	want, ok := builtinActivations[name]
	if !ok {
		return false
	}
	fn := ActivationRegistry.Get(name)
	return fn != nil && reflect.ValueOf(fn).Pointer() == want
}

// CheckBatch validates the shape every network and input of a batch must share,
// for packages that implement a batch themselves. It returns the number of
// outputs the networks share, or an error if any network is recurrent or they
// do not agree with each other or with the inputs.
func CheckBatch(nets []*Network, inputs [][]float64) (numOutputs int, err error) {
	for i, net := range nets {
		if net.recurrent {
			return 0, fmt.Errorf("%w: network %d", ErrNotFeedForward, i)
		}
		if i == 0 {
			numOutputs = len(net.outputs)
		} else if len(net.outputs) != numOutputs {
			return 0, fmt.Errorf("%w: network %d produces %d outputs, network 0 produces %d", ErrOutputSize, i, len(net.outputs), numOutputs)
		}
		for s, input := range inputs {
			if len(input) != net.numInputs {
				return 0, fmt.Errorf("%w: network %d expects %d inputs, sample %d has %d", ErrInputSize, i, net.numInputs, s, len(input))
			}
		}
	}
	return numOutputs, nil
}
