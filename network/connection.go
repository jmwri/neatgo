package network

func NewConnection(id, from, to int, weight float64, enabled bool) Connection {
	return Connection{
		ID:      id,
		From:    from,
		To:      to,
		Weight:  weight,
		Enabled: enabled,
	}
}

type Connection struct {
	ID       int
	From, To int
	Weight   float64
	Enabled  bool
}
