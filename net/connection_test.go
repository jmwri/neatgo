package net_test

import (
	"neatgo/net"
	"testing"
)

func TestConnection_Activate(t *testing.T) {
	t.Parallel()
	c := net.NewConnection(123, 0.5, 1, 2)
	c.SetInput(0.4)
	c.Activate()
	out := c.Output()
	if out != 0.2 {
		t.Errorf("expected %f, got %f", 0.2, out)
	}
}
