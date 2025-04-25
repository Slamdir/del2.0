with Del.Operators;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Ada.Numerics;
With Ada.Float_Text_IO;
package body Del.Loss is

   package ANEF renames Ada.Numerics.Elementary_Functions;

   overriding function Forward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Float is
      Epsilon     : constant Float := 1.0E-8;
      Total_Loss  : Float := 0.0;

      Rows        : constant Integer := Shape(Expected)(1);
      Columns     : constant Integer := Shape(Expected)(2);
   begin

      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Proba_Element     : Element_T := Actual([I, J]);
               begin
                  --  Total_Loss := Total_Loss - (Float(Expected_Element) * ANEF.Log(Float(Proba_Element) + Epsilon) + (1.0 - Float(Expected_Element)) * ANEF.Log(1.0 - Float(Proba_Element) + Epsilon));
                  Total_Loss := Total_Loss - Float(Expected_Element) * ANEF.Log(Float(Proba_Element) + Epsilon);
               end;
         end loop;
      end loop;

      Total_Loss := Total_Loss / Float(Rows);
      return Total_Loss;
   end Forward;

   overriding function Backward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T is
      Gradient    : Tensor_T := Zeros(Actual.Shape);

      Rows        : Integer := Shape(Actual)(1);
   begin

      Gradient := Actual - Expected;
      Gradient := Gradient / Element_T(Rows);
      return Gradient;
   end Backward;

   overriding function Forward (L : Mean_Square_Error_T; Expected, Actual : Tensor_T) return Float is
      Total_Loss : Float := 0.0;
      Rows        : constant Integer := Shape(Expected)(1);
      Columns     : constant Integer := Shape(Expected)(2);
   begin

      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Actual_Element    : Element_T := Actual([I, J]);
               begin
                  Total_Loss := Total_Loss + ( (Float(Expected_Element) - Float(Actual_Element)) ** 2);
               end;
         end loop;
      end loop;

      Total_Loss := Total_Loss / (Float(Rows) * Float(Columns) );
      return Total_Loss;
   end Forward;

   overriding function Backward (L : Mean_Square_Error_T; Expected, Actual : Tensor_T) return Tensor_T is
      Gradient    : Tensor_T := Zeros(Expected.Shape);
      Rows        : constant Integer := Shape(Expected)(1);
      Columns     : constant Integer := Shape(Expected)(2);
   begin
   
      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Actual_Element    : Element_T := Actual([I, J]);
               begin
                  Gradient.Set((I, J), Element_T((-2.0 * (Float(Expected_Element) - Float(Actual_Element))) / (Float(Rows) * Float(Columns))) );
               end;
         end loop;
      end loop;

      return Gradient;
   end Backward;

   -- Implementation of Mean Absolute Error (MAE) loss function
   overriding function Forward (L : Mean_Absolute_Error_T; Expected, Actual : Tensor_T) return Float is
      Total_Loss : Float := 0.0;
      Rows       : constant Integer := Shape(Expected)(1);
      Columns    : constant Integer := Shape(Expected)(2);
   begin
      -- Calculate the sum of absolute differences between expected and actual values
      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element : Element_T := Expected([I, J]);
               Actual_Element   : Element_T := Actual([I, J]);
               Diff            : Float := Float(Expected_Element) - Float(Actual_Element);
            begin
               -- Add absolute difference to total loss
               Total_Loss := Total_Loss + abs(Diff);
            end;
         end loop;
      end loop;

      -- Return the mean absolute error
      Total_Loss := Total_Loss / (Float(Rows) * Float(Columns));
      return Total_Loss;
   end Forward;

   overriding function Backward (L : Mean_Absolute_Error_T; Expected, Actual : Tensor_T) return Tensor_T is
      Gradient    : Tensor_T := Zeros(Expected.Shape);
      Rows        : constant Integer := Shape(Expected)(1);
      Columns     : constant Integer := Shape(Expected)(2);
      Scale_Factor : constant Float := 1.0 / (Float(Rows) * Float(Columns));
   begin
      -- Calculate the gradient of MAE: sign(actual - expected) / (rows * columns)
      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element : Element_T := Expected([I, J]);
               Actual_Element   : Element_T := Actual([I, J]);
               Diff            : Float := Float(Actual_Element) - Float(Expected_Element);
               Grad_Value      : Float := 0.0;
            begin
               -- Gradient is sign(actual - expected) scaled by factor
               if Diff > 0.0 then
                  Grad_Value := Scale_Factor;
               elsif Diff < 0.0 then
                  Grad_Value := -Scale_Factor;
               else
                  Grad_Value := 0.0; -- Zero gradient at exact match points
               end if;
               
               Gradient.Set((I, J), Element_T(Grad_Value));
            end;
         end loop;
      end loop;

      return Gradient;
   end Backward;

end Del.Loss;