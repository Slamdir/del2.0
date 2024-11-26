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

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

end Del.Model;