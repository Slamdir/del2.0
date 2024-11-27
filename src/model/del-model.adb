package body Del.Model is
    procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
    begin
        Self.Layers.Append(Layer);
    end Add_Layer;

    function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T is
        Current : Tensor_T := Input;
    begin
        -- Pass data through each layer
        for Layer of Self.Layers loop
            Current := Layer.Forward(Current);
        end loop;
        return Current;
    end Run_Layers;
end Del.Model;