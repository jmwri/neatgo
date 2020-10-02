package net

import "neatgo"

type BasicConnection interface {
	neatgo.Identifier
	From() int64
	To() int64
}

type InConnection interface {
	BasicConnection
	Outputter
}

type OutConnection interface {
	BasicConnection
	Activator
	InputSetter
}

func NewConnection(id int64, weight float64, from int64, to int64) *Connection {
	return &Connection{
		id:     id,
		from:   from,
		to:     to,
		input:  0,
		weight: weight,
		output: 0,
	}
}

type Connection struct {
	id     int64
	from   int64
	to     int64
	input  float64
	weight float64
	output float64
}

func (c *Connection) ID() int64 {
	return c.id
}

func (c *Connection) From() int64 {
	return c.from
}

func (c *Connection) To() int64 {
	return c.to
}

func (c *Connection) SetInput(v float64) {
	c.input = v
}

func (c *Connection) Activate() {
	c.output = c.input * c.weight
}

func (c *Connection) Output() float64 {
	return c.output
}

func NewConnectionDefinition(from Sender, to Receiver, weight float64) *ConnectionDefinition {
	return &ConnectionDefinition{
		From:   from,
		To:     to,
		Weight: weight,
	}
}

type ConnectionDefinition struct {
	From   Sender
	To     Receiver
	Weight float64
}

func AddConnections(definitions []*ConnectionDefinition) {
	for i, pair := range definitions {
		c := NewConnection(int64(i+1), pair.Weight, pair.From.ID(), pair.To.ID())
		pair.From.AddOutConnection(c)
		pair.To.AddInConnection(c)
	}
}
