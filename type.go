package pytorch

// #include "cbits/predictor.hpp"
import "C"
import (
	"reflect"

	"gorgonia.org/tensor"
)

// DType tensor scalar data type
type DType C.Torch_DataType

var types = []struct {
	typ      reflect.Type
	dataType C.Torch_DataType
}{
	{reflect.TypeOf(uint8(0)), C.Torch_Byte},
	{reflect.TypeOf(int8(0)), C.Torch_Char},
	{reflect.TypeOf(int16(0)), C.Torch_Short},
	{reflect.TypeOf(int32(0)), C.Torch_Int},
	{reflect.TypeOf(int64(0)), C.Torch_Long},
	// Go doesn't have single precision floating point
	// {reflect.TypeOf(float16(0)), C.Torch_Half},
	{reflect.TypeOf(float32(0)), C.Torch_Float},
	{reflect.TypeOf(float64(0)), C.Torch_Double},
}

func fromType(ten *tensor.Dense) DType {
	for _, t := range types {
		if t.typ == ten.Dtype().Type {
			return DType(t.dataType)
		}
	}
	return C.Torch_Unknown
}
