with Orka.Numerics.Singles.Tensors;
with Ada.Text_IO; use Ada.Text_IO;
with Del.Operators;  -- Added to bring layer types into scope
with Ada.Tags; use Ada.Tags;  -- Added to allow comparison of tags

package body Del.Model is

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
   begin
       Self.Layers.Append(Layer);
   end Add_Layer;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

   -- Function to get layer name (example: add similar Get_Name function for each type)
   function Get_Layer_Name(Layer : Func_Access_T) return String is
   begin
      if Layer'Tag = Del.Operators.Linear_T'Tag then
         return "Linear Layer";
      elsif Layer'Tag = Del.Operators.ReLU_T'Tag then
         return "ReLU Layer";
      elsif Layer'Tag = Del.Operators.SoftMax_T'Tag then
         return "SoftMax Layer";
      else
         return "Unknown Layer";
      end if;
   end Get_Layer_Name;

   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T is
      Current : Tensor_T := Input;
   begin
       -- Pass data through each layer
       for Layer of Self.Layers loop
           Put_Line("Running layer of type: " & Get_Layer_Name(Layer));
           Current := Layer.Forward(Current);
           Put_Line("Output Tensor Shape after this layer:");
           Put_Line("Tensor Shape: " & Shape(Current)'Image);
       end loop;
       return Current;
   end Run_Layers;

end Del.Model;
