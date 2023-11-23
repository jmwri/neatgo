package network

import "context"

type Network interface {
	Activate(ctx context.Context, in ...float64) error
	Output() []float64
}
