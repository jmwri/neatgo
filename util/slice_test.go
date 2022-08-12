package util_test

import (
	"github.com/jmwri/neatgo/util"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRemoveSliceIndex(t *testing.T) {
	s := []int{1, 2, 3}
	actual := util.RemoveSliceIndex(s, 1)
	assert.Equal(t, actual, []int{1, 3})
	actual = util.RemoveSliceIndex(actual, 1)
	assert.Equal(t, actual, []int{1})
	actual = util.RemoveSliceIndex(actual, 0)
	assert.Equal(t, actual, []int{})
}

func TestInSlice(t *testing.T) {
	s := []int{1, 2, 3}
	assert.True(t, util.InSlice(s, 1))
	assert.True(t, util.InSlice(s, 2))
	assert.True(t, util.InSlice(s, 3))
	assert.False(t, util.InSlice(s, 5))
}

func TestRandSliceElement(t *testing.T) {
	s := []int{1, 2, 3}

	seenMap := make(map[int]int)

	for i := 1; i < 100; i++ {
		choice := util.RandSliceElement(s)
		seenMap[choice] += 1
	}

	for _, numSeen := range seenMap {
		assert.Greater(t, numSeen, 0)
	}
}
