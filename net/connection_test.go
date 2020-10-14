package net_test

import (
	"github.com/jmwri/neatgo/net"
	"reflect"
	"testing"
)

func TestConnection_Copy(t *testing.T) {
	c1 := net.NewConnection(1, .5, 2, 3, true)
	c2 := c1.Copy()
	c2.SetWeight(.6)

	if c1 == c2 {
		t.Fatal("pointer to the same node")
	}
	if reflect.DeepEqual(c1, c2) {
		t.Fatal("connections are the same")
	}
	if c1.Weight() == c2.Weight() {
		t.Fatal("weights are the same")
	}
}
