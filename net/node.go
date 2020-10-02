package net

import (
	"neatgo"
	"neatgo/activation"
)

type Activator interface {
	Activate()
}

type Outputter interface {
	Output() float64
}

type ActivationFnRunner interface {
	SetActivationFn(fn activation.Fn)
}

type Receiver interface {
	neatgo.Identifier
	AddInConnection(c InConnection)
	Inputs() []InConnection
}

type Sender interface {
	neatgo.Identifier
	AddOutConnection(c OutConnection)
	Outputs() []OutConnection
}

type InputSetter interface {
	SetInput(v float64)
}

type BiasNode interface {
	Node
	Sender
	SetBias(v float64)
}

type InputNode interface {
	Node
	Sender
	InputSetter
}

type HiddenNode interface {
	Node
	ActivationFnRunner
	Receiver
	Sender
}

type OutputNode interface {
	Node
	ActivationFnRunner
	Receiver
}

type Node interface {
	neatgo.Identifier
	Activator
	Outputter
}

func NewBiasNode(id int64, input float64) *biasNode {
	return &biasNode{
		id:      id,
		input:   input,
		outputs: make([]OutConnection, 0),
	}
}

type biasNode struct {
	id      int64
	input   float64
	outputs []OutConnection
}

func (n *biasNode) ID() int64 {
	return n.id
}

func (n *biasNode) SetBias(v float64) {
	n.input = v
}

func (n *biasNode) Activate() {
	for _, out := range n.outputs {
		out.SetInput(n.input)
		out.Activate()
	}
}

func (n *biasNode) Outputs() []OutConnection {
	return n.outputs
}

func (n *biasNode) AddOutConnection(c OutConnection) {
	n.outputs = append(n.outputs, c)
}

func (n *biasNode) Output() float64 {
	return n.input
}

func NewInputNode(id int64) *inputNode {
	return &inputNode{
		id:      id,
		input:   0,
		outputs: make([]OutConnection, 0),
	}
}

type inputNode struct {
	id      int64
	input   float64
	outputs []OutConnection
}

func (n *inputNode) ID() int64 {
	return n.id
}

func (n *inputNode) SetInput(v float64) {
	n.input = v
}

func (n *inputNode) Activate() {
	for _, out := range n.outputs {
		out.SetInput(n.input)
		out.Activate()
	}
}

func (n *inputNode) Output() float64 {
	return n.input
}

func (n *inputNode) AddOutConnection(c OutConnection) {
	n.outputs = append(n.outputs, c)
}

func (n *inputNode) Outputs() []OutConnection {
	return n.outputs
}

func NewHiddenNode(id int64, activationFn activation.Fn) *hiddenNode {
	return &hiddenNode{
		id:           id,
		inputs:       make([]InConnection, 0),
		outputs:      make([]OutConnection, 0),
		activationFn: activationFn,
		output:       0,
	}
}

type hiddenNode struct {
	id           int64
	inputs       []InConnection
	outputs      []OutConnection
	activationFn activation.Fn
	output       float64
}

func (n *hiddenNode) ID() int64 {
	return n.id
}

func (n *hiddenNode) Activate() {
	sum := 0.0
	for _, inputConn := range n.inputs {
		sum += inputConn.Output()
	}

	activated := n.activationFn(sum)
	n.output = activated

	for _, out := range n.outputs {
		out.SetInput(n.output)
		out.Activate()
	}
}

func (n *hiddenNode) AddOutConnection(c OutConnection) {
	n.outputs = append(n.outputs, c)
}

func (n *hiddenNode) Output() float64 {
	return n.output
}

func (n *hiddenNode) AddInConnection(c InConnection) {
	n.inputs = append(n.inputs, c)
}

func (n *hiddenNode) Inputs() []InConnection {
	return n.inputs
}

func (n *hiddenNode) Outputs() []OutConnection {
	return n.outputs
}

func (n *hiddenNode) SetActivationFn(fn activation.Fn) {
	n.activationFn = fn
}

func NewOutputNode(id int64, activationFn activation.Fn) *outputNode {
	return &outputNode{
		id:           id,
		inputs:       make([]InConnection, 0),
		activationFn: activationFn,
		output:       0,
	}
}

type outputNode struct {
	id           int64
	inputs       []InConnection
	activationFn activation.Fn
	output       float64
}

func (n *outputNode) ID() int64 {
	return n.id
}

func (n *outputNode) Activate() {
	sum := 0.0
	for _, inputConn := range n.inputs {
		sum += inputConn.Output()
	}

	activated := n.activationFn(sum)
	n.output = activated
}

func (n *outputNode) Output() float64 {
	return n.output
}

func (n *outputNode) AddInConnection(c InConnection) {
	n.inputs = append(n.inputs, c)
}

func (n *outputNode) Inputs() []InConnection {
	return n.inputs
}

func (n *outputNode) SetActivationFn(fn activation.Fn) {
	n.activationFn = fn
}
