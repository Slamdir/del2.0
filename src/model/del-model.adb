package body Del.Model is

    procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is

    begin
        Self.Layers.Append(Layer);
    end Add_Layer;

    procedure Run_Layers(Self : in Model) is

    begin
        for E of Self.Layers loop
            declare 
                T : Tensor_T := E.all.Forward(Zeros((2, 2)));
            begin
                null;
            end;
        end loop;
    end Run_Layers;

end Del.Model;