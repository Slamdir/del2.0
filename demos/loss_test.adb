with Ada.Text_IO; use Ada.Text_IO;
with Ada.Float_Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Del;
with Del.Loss;

procedure Loss_Test is

   package D     renames Del;
   package DLoss renames Del.Loss;

   -- Expected and Actual tensors for testing the cross-entropy loss
   Expected : constant D.Tensor_T := To_Tensor (
      [1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0], [3, 3]);

   Actual : constant D.Tensor_T := To_Tensor (
      [2.0, 1.0, 0.1,
       0.5, 2.0, 0.5,
       0.3, 0.2, 3.0], [3, 3]);

   -- Instantiate the Cross_Entropy_T loss object
   Cross_Entropy : DLoss.Cross_Entropy_T;

   -- Forward and backward loss outputs
   Loss_Value : D.Element_T;
   Gradient   : D.Tensor_T := Zeros (Actual.Shape);  -- Initialized properly

   -- Procedure to print Tensor_T
   procedure Print_Tensor (T : D.Tensor_T) is
      use Ada.Float_Text_IO;

      Rows  : constant Positive := T.Shape (1);
      Cols  : constant Positive := T.Shape (2);
   begin
      Put_Line ("[");
      for I in 1 .. Rows loop
         Put (" [");
         declare
            Row_Slice : constant D.Tensor_T := T (I);  -- Assuming slicing is possible with `T(I)`
         begin
            for J in 1 .. Cols loop
               declare
                  -- Storing in an intermediate variable to resolve ambiguity
                  Element_F32 : constant Float_32 := Row_Slice.Get (J);  -- Get element of type Float_32
                  Element     : constant Standard.Float := Standard.Float (Element_F32);  -- Convert explicitly to Standard.Float
               begin
                  Put (Item => Element, Aft => 6, Exp => 0);
                  if J < Cols then
                     Put (", ");
                  end if;
               end;
            end loop;
         end;
         Put ("]");
         if I < Rows then
            Put_Line (",");
         else
            Put_Line ("");
         end if;
      end loop;
      Put_Line ("]");
   end Print_Tensor;

begin
   Put_Line ("Testing Del.Loss Package");
   Put_Line ("--------------------------------");

   -- Test Cross_Entropy_T Forward
   Put_Line ("Testing Cross_Entropy_T Forward:");
   Loss_Value := DLoss.Forward (Cross_Entropy, Expected, Actual);
   Put_Line ("Cross-Entropy Loss Value: " & Float'Image (Float (Loss_Value)));
   Put_Line ("");

   -- Test Cross_Entropy_T Backward
   Put_Line ("Testing Cross_Entropy_T Backward:");
   Gradient := DLoss.Backward (Cross_Entropy, Expected, Actual);
   Put_Line ("Cross-Entropy Gradient:");
   Print_Tensor (Gradient);
   Put_Line ("");

end Loss_Test;
