with Ada.Text_IO; use Ada.Text_IO;
with Ada.Containers; use Ada.Containers;
with Ada.Exceptions;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;

package body Del.Model is
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
   begin
       Self.Layers.Append(Layer);
   end Add_Layer;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T is
   begin
      Put_Line("Run_Layers called with input shape: " & 
               Shape(Input)(1)'Image & "," & Shape(Input)(2)'Image);
               
      if Self.Layers.Length = 0 then
          Put_Line("No layers in network");
          return Input;
      end if;

      Put_Line("Network has" & Self.Layers.Length'Image & " layers");

      declare
         Current_Input : Tensor_T := Input;
         First_Layer  : constant Func_Access_T := Self.Layers.First_Element;
         First_Output : constant Tensor_T := First_Layer.all.Forward(Current_Input);
         Result : Tensor_T := First_Output;
         C : Layer_Vectors.Cursor := Self.Layers.First;
      begin
         -- Skip the first element since we've already processed it
         Layer_Vectors.Next(C);
         
         while Layer_Vectors.Has_Element(C) loop
             Put_Line("Processing next layer");
             Put_Line("Current input shape: " & Shape(Result)(1)'Image & "," & Shape(Result)(2)'Image);
             
             declare
                Current_Layer : constant Func_Access_T := Layer_Vectors.Element(C);
             begin
                Result := Current_Layer.all.Forward(Result);
             end;
             
             Put_Line("Layer output shape: " & Shape(Result)(1)'Image & "," & Shape(Result)(2)'Image);
             Layer_Vectors.Next(C);
         end loop;
         
         return Result;
      end;
   exception
      when E : others =>
         Put_Line("Error in Run_Layers: ");
         Put_Line(Ada.Exceptions.Exception_Information(E));
         raise;
   end Run_Layers;
end Del.Model;