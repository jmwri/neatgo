package feedforward

import "github.com/jmwri/neatgo/util"

type Connection struct {
	id, from, to int64
	weight       float64
	enabled      bool
	value        float64
}

func (c Connection) ID() int64 {
	return c.id
}

func (c Connection) From() int64 {
	return c.from
}

func (c Connection) To() int64 {
	return c.to
}

func (c Connection) Enabled() bool {
	return c.enabled
}

func (c Connection) Value() float64 {
	return c.value
}

func (c Connection) Activate(inputs ...float64) Connection {
	sum := util.Sum(inputs...)
	c.value = sum * c.weight
	return c
}
