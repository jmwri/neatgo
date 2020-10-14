package util_test

import (
	"fmt"
	"github.com/jmwri/neatgo/util"
	"testing"
)

func TestSliceOfFloatEqual(t *testing.T) {
	tests := []struct {
		a        []float64
		b        []float64
		expected bool
	}{
		{a: []float64{1, 2, 3, 4, 5}, b: []float64{1, 2, 3, 4, 5}, expected: true},
		{a: []float64{2, 1, 3, 4, 5}, b: []float64{1, 2, 3, 4, 5}, expected: false},
	}
	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v == %v", testCase.a, testCase.b), func(t *testing.T) {
			res := util.SliceOfFloatEqual(testCase.a, testCase.b)

			if res != testCase.expected {
				t.Errorf("expected %v, got %v", testCase.expected, res)
			}
		})
	}
}

func TestSliceOfInt64Equal(t *testing.T) {
	tests := []struct {
		a        []int64
		b        []int64
		expected bool
	}{
		{a: []int64{1, 2, 3, 4, 5}, b: []int64{1, 2, 3, 4, 5}, expected: true},
		{a: []int64{2, 1, 3, 4, 5}, b: []int64{1, 2, 3, 4, 5}, expected: false},
	}
	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v == %v", testCase.a, testCase.b), func(t *testing.T) {
			res := util.SliceOfInt64Equal(testCase.a, testCase.b)

			if res != testCase.expected {
				t.Errorf("expected %v, got %v", testCase.expected, res)
			}
		})
	}
}

func TestSortSliceOfFloatAsc(t *testing.T) {
	tests := []struct {
		input    []float64
		expected []float64
	}{
		{input: []float64{4, 2, 5, 1, 3}, expected: []float64{1, 2, 3, 4, 5}},
		{input: []float64{4, 3, 4, 1, 5, 2, 1.2, 1.3}, expected: []float64{1, 1.2, 1.3, 2, 3, 4, 4, 5}},
	}
	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v", testCase.input), func(t *testing.T) {
			util.SortSliceOfFloatAsc(testCase.input)
			res := util.SliceOfFloatEqual(testCase.input, testCase.expected)

			if res != true {
				t.Errorf("expected %v, got %v", testCase.expected, testCase.input)
			}
		})
	}
}

func TestRemoveInt64FromSlice(t *testing.T) {
	tests := []struct {
		input    []int64
		remove   int64
		expected []int64
	}{
		{input: []int64{4, 2, 5, 1, 3}, remove: 5, expected: []int64{4, 2, 1, 3}},
		{input: []int64{1, 2, 2, 2, 3}, remove: 2, expected: []int64{1, 2, 2, 3}},
		{input: []int64{1, 2, 2, 2, 3}, remove: 5, expected: []int64{1, 2, 2, 2, 3}},
	}
	for _, testCase := range tests {
		testCase := testCase
		t.Run(fmt.Sprintf("%v - %d", testCase.input, testCase.remove), func(t *testing.T) {
			res := util.RemoveInt64FromSlice(testCase.input, testCase.remove)

			if !util.SliceOfInt64Equal(res, testCase.expected) {
				t.Errorf("expected %v, got %v", testCase.expected, res)
			}
		})
	}
}
