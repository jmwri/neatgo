package net

func NewConnection(id int64, weight float64, from int64, to int64, enabled bool) *Connection {
	return &Connection{
		id:      id,
		from:    from,
		to:      to,
		weight:  weight,
		enabled: enabled,
	}
}

type Connection struct {
	id      int64
	from    int64
	to      int64
	weight  float64
	enabled bool
}

func (c *Connection) Copy() *Connection {
	return &Connection{
		id:      c.id,
		from:    c.from,
		to:      c.to,
		weight:  c.weight,
		enabled: c.enabled,
	}
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

func (c *Connection) Weight() float64 {
	return c.weight
}

func (c *Connection) SetWeight(w float64) {
	c.weight = w
}

func (c *Connection) Enabled() bool {
	return c.enabled
}

func (c *Connection) SetEnabled(e bool) {
	c.enabled = e
}
