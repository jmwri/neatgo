package net

func NewConnection(id int64, weight float64, from int64, to int64) *Connection {
	return &Connection{
		id:     id,
		from:   from,
		to:     to,
		weight: weight,
	}
}

type Connection struct {
	id     int64
	from   int64
	to     int64
	weight float64
}

func (c *Connection) Copy() *Connection {
	return &Connection{
		id:     c.id,
		from:   c.from,
		to:     c.to,
		weight: c.weight,
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
