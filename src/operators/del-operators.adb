with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics;

package body Del.Operators is

   overriding function Forward (L : Linear_T; X : Tensor_T) return Tensor_T is
   begin
      Put_Line("Forward from Linear_T");
      return X;
      --(X * L.Map ("Weights") + L.Map ("Bias"));
   end Forward;
   
   overriding function Backward (L : Linear_T; Dy : Tensor_T) return Tensor_T is
   begin
      return Dy;
   end Backward;


   overriding function Forward (L : ReLU_T; X : Tensor_T) return Tensor_T is
   begin
      Put_Line ("Forward from ReLu_T");
      return X;
   end Forward;

   overriding function Backward (L : ReLU_T; Dy : Tensor_T) return Tensor_T is
   begin
      return dY;
   end Backward;


   function Row_Sum(Values : Tensor_T) return Tensor_T is
      Rows     : Integer := Shape(Values)(1);
      Output   : Tensor_T := Zeros((Rows, 1));
   begin
      --  Put_Line ("Rows: " & Rows'Image & " Columns: " & Columns'Image);
      for I in 1 .. Rows loop
      declare
         Row_I : Tensor_T := Values(I);
         begin
            Output.Set(I, Sum(Row_I));
         end;
      end loop;
      --  Put_Line(Image(Output));
      return Output;
   end Row_Sum;


   -- Allows for Cross_Entropy to call SoftMax with the Actual Values
   function SoftMax(X : Tensor_T) return Tensor_T is

      function Divide_By_Row (Exp_Values : Tensor_T; Exp_Values_Sum : Tensor_T) return Tensor_T is
         Rows     : Integer := Shape(Exp_Values)(1);
         Output   : Tensor_T := Zeros(Shape(Exp_Values));
      begin
         for I in 1 .. Rows loop
         declare
            Exp_Values_Row_I     : Tensor_T := Exp_Values(I);
            Exp_Values_Element_I : Element_T := Exp_Values_Sum(I);
            begin
               Output.Set(I, Exp_Values_Row_I / Exp_Values_Element_I);
            end;
         end loop;
         return Output;
      end Divide_By_Row;

      Exp_Values     : Tensor_T := Ada.Numerics.e ** X;
      Exp_Values_Sum : Tensor_T := Row_Sum(Exp_Values);
      Output         : Tensor_T := Divide_By_Row(Exp_Values, Exp_Values_Sum);

   begin
      return Output;
   end SoftMax;

   -- Acts as a proxy to call SoftMax
   overriding function Forward (L : SoftMax_T; X : Tensor_T) return Tensor_T is
      Output : Tensor_T := Softmax(X);
   begin
      return Output;
   end Forward;

   -- This should only be called after Cross-Entropy
   overriding function Backward (L : SoftMax_T; Dy : Tensor_T) return Tensor_T is
      --  I : Tensor_T := Identity (Shape(Dy)(1));
      --  Output : Tensor_T := Forward(L, Dy) * (I - Forward(L, Dy));
   begin
      return dY;
   end Backward;

end Del.Operators;