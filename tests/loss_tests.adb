with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Exceptions;            use Ada.Exceptions;
with Ada.Directories;           use Ada.Directories;

with Del;                       use Del;
with Del.Loss;                  -- For Cross_Entropy_T
with Del.Model;
with Del.Operators;
with Orka.Numerics.Singles.Tensors;     use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Loss_Tests is
begin
   Put_Line("Program started");
   Put_Line("=== Loss Function Testing ===");

   declare
      My_Model       : Del.Model.Model;
      Loss_Function  : Del.Loss.Cross_Entropy_T;
      Data_Shape     : constant Tensor_Shape_T := (1 => 1, 2 => 3);  -- One-hot labels, 3 classes
      Target_Shape   : constant Tensor_Shape_T := (1 => 1, 2 => 3);  -- Matches Data shape
      Minimal_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 1);
      Empty_Tensor   : constant Tensor_T       := Empty(Minimal_Shape);
      Expected       : Tensor_T                := Zeros(Data_Shape); -- Expected class probabilities
      Actual         : Tensor_T                := Zeros(Data_Shape); -- Logits before softmax
   begin
      Put_Line("Variables declared");

      -- Manually define an example for Forward and Backward pass
      Put_Line("Defining Expected (one-hot label) and Actual (logits) tensors...");

      -- Expected (one-hot encoding): Class 2 is the correct class
      Expected.Set((1,1), 0.0);
      Expected.Set((1,2), 1.0);
      Expected.Set((1,3), 0.0);

      -- Actual (logits before softmax)
      Actual.Set((1,1), 1.0);
      Actual.Set((1,2), 2.0);
      Actual.Set((1,3), 3.0);

      declare
         Computed_Loss : Element_T;
      begin
         Put_Line("Testing Forward Pass (Cross Entropy Loss Calculation)...");
         Computed_Loss := Loss_Function.Forward(Expected, Actual);

         -- Expected loss ≈ 1.41, so we check within a small tolerance
         if Float(Computed_Loss) > 0.0 then
            if abs(Float(Computed_Loss) - 1.41) < 0.05 then
               Put_Line("PASS: Cross-entropy loss ~ " 
                 & Float(Computed_Loss)'Img & " (Expected ~1.41)");
            else
               Put_Line("FAIL: Cross-entropy loss = " 
                 & Float(Computed_Loss)'Img & ", expected ~1.41");
            end if;
         else
            Put_Line("FAIL: Computed loss is non-positive: " 
              & Float(Computed_Loss)'Img);
         end if;
      exception
         when E : others =>
            Put_Line("FAIL: Exception in Forward pass: " 
              & Exception_Message(E));
      end;

      -- Backward Pass (Gradient Calculation)
      declare
         Gradient : Tensor_T := Zeros(Data_Shape);  -- Fixed initialization
         G1, G2, G3 : Float;
      begin
         Put_Line("Testing Backward Pass (Gradient Calculation)...");
         Gradient := Loss_Function.Backward(Expected, Actual);

         G1 := Float(Element_T(Gradient.Get((1,1))));
         G2 := Float(Element_T(Gradient.Get((1,2))));
         G3 := Float(Element_T(Gradient.Get((1,3))));

         Put_Line("Gradient = [" & G1'Img & ", " & G2'Img & ", " & G3'Img & "]");

         -- Expected gradient = Softmax(Actual) - Expected
         -- Approximated: [0.09, -0.7553, 0.6653]
         if    abs(G1 - 0.09)     < 0.02
            and abs(G2 + 0.7553) < 0.02
            and abs(G3 - 0.6653) < 0.02 then
            Put_Line("PASS: Cross-entropy backward gradient matches expected values.");
         else
            Put_Line("FAIL: Cross-entropy backward gradient out of expected range.");
         end if;
      exception
         when E : others =>
            Put_Line("FAIL: Exception in Backward pass: " 
              & Exception_Message(E));
      end;

      Put_Line("Loss function testing completed successfully!");
   end;

exception
   when E : others =>
      Put_Line("Unexpected error: " & Exception_Message(E));
end Loss_Tests;
