package net_test

import (
	"neatgo/activation"
	"neatgo/aggregation"
	"neatgo/net"
	"reflect"
	"testing"
)

func TestNode_Copy(t *testing.T) {
	n1 := net.NewNode(1, 0, activation.Nil, aggregation.Sum)
	n2 := n1.Copy()
	n1.SetActivationFn(activation.Sigmoid)

	if n1 == n2 {
		t.Fatal("pointer to the same node")
	}
	if reflect.DeepEqual(n1, n2) {
		t.Fatal("nodes are the same")
	}
}
