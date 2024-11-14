with Del.Operators;

package body Del.Operators is

    --Linear
    overriding function Forward (Self : Linear_T; X : Tensor_T) return Tensor_T is
        --This should be an initialized Tensor of shape ((X -> Row Size) x (Self.Element("Weights") -> Columns Size))
        --IE: X = (3x1) and Self.Weights = (1x3) => Output = (3x3)
        Output : Tensor_T := Zeros((2, 2));
    begin
        --Output := (X * Self.Element("Weights")) + Self.Element("Bias");
        --Self.Element("Input")
        return Output;
    end;

    overriding function Backward (Self : Linear_T; X : Tensor_T) return Tensor_T is

    begin
        return Data : Tensor_T := Zeros((2, 2));
    end;

    --ReLU
    overriding function Forward (Self : ReLU_T; X : Tensor_T) return Tensor_T is

    begin
        return Data : Tensor_T := Zeros((2, 2));
    end;

    overriding function Backward (Self : ReLU_T; X : Tensor_T) return Tensor_T is

    begin
        return Data : Tensor_T := Zeros((2, 2));
    end;

end Del.Operators;