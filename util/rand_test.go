package util_test

import (
	"fmt"
	"github.com/jmwri/neatgo/util"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFloatBetween(t *testing.T) {
	type testCase struct {
		min, max float64
	}
	tests := []testCase{
		{
			min: 0,
			max: 100,
		},
		{
			min: 0,
			max: 1,
		},
		{
			min: 5,
			max: 6,
		},
		{
			min: 5,
			max: 5.2,
		},
		{
			min: 9999,
			max: 10000,
		},
		{
			min: -10,
			max: 0,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			for i := 1; i < 20; i++ {
				actual := util.FloatBetween(test.min, test.max)
				assert.GreaterOrEqual(t, actual, test.min)
				assert.LessOrEqual(t, actual, test.max)
			}
		})
	}
}

func TestIntBetween(t *testing.T) {
	type testCase struct {
		min, max int
	}
	tests := []testCase{
		{
			min: 0,
			max: 100,
		},
		{
			min: 0,
			max: 1,
		},
		{
			min: 5,
			max: 6,
		},
		{
			min: 9999,
			max: 10000,
		},
		{
			min: -10,
			max: 0,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			for i := 1; i < 20; i++ {
				actual := util.IntBetween(test.min, test.max)
				assert.GreaterOrEqual(t, actual, test.min)
				assert.LessOrEqual(t, actual, test.max)
			}
		})
	}
}
