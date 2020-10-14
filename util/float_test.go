package util_test

import (
	"fmt"
	"github.com/jmwri/neatgo/util"
	"math/rand"
	"testing"
)

func TestRandFloat(t *testing.T) {
	rand.Seed(12345)
	tests := []struct {
		min      float64
		max      float64
		expected float64
	}{
		{min: 0, max: 1, expected: .8487305991992138},
		{min: 0, max: 1, expected: .6451080292174168},
		{min: 0, max: 1, expected: .7382079884862905},
		{min: 0, max: 1, expected: .31522206779732853},
		{min: 0, max: 1, expected: .057001989921077224},
	}

	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%f - %f", testCase.min, testCase.max), func(t *testing.T) {
			res := util.RandFloat(testCase.min, testCase.max)
			if res != testCase.expected {
				t.Errorf("expected %f, got %f", testCase.expected, res)
			}
		})
	}
}

func TestReduceFloat(t *testing.T) {
	tests := []struct {
		init     float64
		sequence []float64
		expected float64
	}{
		{0, []float64{2, 3, 2, 4, 5}, 16},
		{5, []float64{2, 3, 2, 4, 5}, 21},
	}

	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%f, %v", testCase.init, testCase.sequence), func(t *testing.T) {
			res := util.ReduceFloat(util.SumFloat, testCase.sequence, testCase.init)

			if res != testCase.expected {
				t.Errorf("expected %f, got %f", testCase.expected, res)
			}
		})
	}
}

func TestMultiplyFloat(t *testing.T) {
	tests := []struct {
		a        float64
		b        float64
		expected float64
	}{
		{5, 3, 15},
		{2, 1, 2},
	}

	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v * %v", testCase.a, testCase.b), func(t *testing.T) {
			res := util.MultiplyFloat(testCase.a, testCase.b)

			if res != testCase.expected {
				t.Errorf("expected %f, got %f", testCase.expected, res)
			}
		})
	}
}

func TestSumFloat(t *testing.T) {
	tests := []struct {
		a        float64
		b        float64
		expected float64
	}{
		{5, 3, 8},
		{2, 1, 3},
	}

	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v + %v", testCase.a, testCase.b), func(t *testing.T) {
			res := util.SumFloat(testCase.a, testCase.b)

			if res != testCase.expected {
				t.Errorf("expected %f, got %f", testCase.expected, res)
			}
		})
	}
}
