with Del.Operators;
with Ada.Numerics.Elementary_Functions;
with Ada.Numerics;
With Ada.Float_Text_IO;
package body Del.Loss is

   overriding function Forward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Element_T is
      Epsilon     : constant Float := 1.0E-8;
      Total_Loss  : Float := 0.0;
      Proba       : constant Tensor_T := Del.Operators.SoftMax(Actual);

      Rows        : constant Integer := Shape(Expected)(1);
      Columns     : constant Integer := Shape(Expected)(2);
   begin

      for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Proba_Element     : Element_T := Proba([I, J]);
               begin
                  Total_Loss := Total_Loss - Float(Expected_Element) * Ada.Numerics.Elementary_Functions.Log(Float(Proba_Element) + Epsilon);
               end;
         end loop;
      end loop;

      return Element_T(Total_Loss);
   end Forward;

   overriding function Backward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T is
      Epsilon     : constant Float := 1.0E-8;
      Gradient    : Tensor_T := Zeros(Actual.Shape);
      Proba       : Tensor_T := Del.Operators.SoftMax(Actual);

      Rows        : Integer := Shape(Actual)(1);
      Columns     : Integer := Shape(Actual)(2);
   begin

   for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Proba_Element     : Element_T := Proba([I, J]);
               Grad_Element      : Float := Float(Proba_Element) - Float(Expected_Element);
               begin
                  Gradient.Set([I, J], Element_T(Grad_Element));
               end;
         end loop;
      end loop;

      return Gradient;
   end Backward;

   overriding function Forward (L : Mean_Square_Error_T; Expected, Actual : Tensor_T) return Element_T is
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
                  Total_Loss := Total_Loss + ((Float(Expected_Element) - Float(Actual_Element)) ** 2);
               end;
         end loop;
      end loop;

      Total_Loss := Total_Loss / (Float(Rows) * Float(Columns));
      return Element_T(Total_Loss);
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

end Del.Loss;