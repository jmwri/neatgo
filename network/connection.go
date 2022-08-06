package network

type Connection struct {
	ID       int
	From, To int
	Weight   float64
}
