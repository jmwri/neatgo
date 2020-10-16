package activation

import (
	"reflect"
)

func IsSameFunction(a, b Fn) bool {
	return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer()
}
