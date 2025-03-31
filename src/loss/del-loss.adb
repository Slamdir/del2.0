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
                  Total_Loss := Total_Loss - (Float(Expected_Element) * ANEF.Log(Float(Proba_Element) + Epsilon) + (1.0 - Float(Expected_Element)) * ANEF.Log(1.0 - Float(Proba_Element) + Epsilon));
                  --  Total_Loss := Total_Loss - Float(Expected_Element) * Ada.Numerics.Elementary_Functions.Log(Float(Proba_Element) + Epsilon);
               end;
         end loop;
      end loop;

      Total_Loss := Total_Loss / Float(Rows);
      return Total_Loss;
   end Forward;

   overriding function Backward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T is
      Epsilon     : constant Float := 1.0E-8;
      Gradient    : Tensor_T := Zeros(Actual.Shape);

      Rows        : Integer := Shape(Actual)(1);
      Columns     : Integer := Shape(Actual)(2);
   begin

   for I in 1 .. Rows loop
         for J in 1 .. Columns loop
            declare
               Expected_Element  : Element_T := Expected([I, J]);
               Proba_Element     : Element_T := Actual([I, J]);
               Grad_Element      : Element_T := Expected_Element - Proba_Element;
               begin
                  Gradient.Set([I, J], Element_T(Grad_Element));
               end;
         end loop;
      end loop;

      New_Line;
      Put_Line("GRADIENT IMAGE:");
      New_Line;
      Put_Line(Gradient.Image);
      New_Line;

      return Gradient;
   end Backward;

end Del.Loss;