with Ada.Text_IO; use Ada.Text_IO;
package body Del.Model is

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
   begin
       Self.Layers.Append(Layer);
   end Add_Layer;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

   procedure Train_Model(Self : in Model; Num_Epochs : Positive; Data : Tensor_T) is

   begin
      for I in 0 .. Num_Epochs loop

         begin
            Put_Line("Num Epcohs: " & I'Image);
         end;
      end loop;
   end Train_Model;

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