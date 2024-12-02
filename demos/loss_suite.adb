with Ada.Text_IO; use Ada.Text_IO;
with Del;
with Del.Operators;
with Del.Loss;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Loss_Suite is
   package D renames Del;
   package DOp renames Del.Operators;
   package DLoss renames Del.Loss;

   -- Test Inputs
   Expected : D.Tensor_T := To_Tensor([1.0, 0.0, 0.0], [1, 3]);  -- One-hot encoded expected output
   Predicted_Correct : D.Tensor_T := To_Tensor([2.0, 1.0, 0.5], [1, 3]); -- Logits favoring the correct class
   Predicted_Incorrect : D.Tensor_T := To_Tensor([0.5, 1.5, 2.0], [1, 3]); -- Logits favoring incorrect class

   -- Expected Outputs
   Expected_Loss_Correct : D.Element_T := D.Element_T(0.40760596); -- Precomputed expected loss
   Expected_Loss_Incorrect : D.Element_T := D.Element_T(2.40760596); -- Precomputed expected loss

   -- Cross-Entropy Loss
   Loss : DLoss.Cross_Entropy_T;

   -- Helper procedure to assert equality
   procedure Assert_Equal(Expected, Actual : D.Element_T; Test_Name : String) is
      Tolerance : constant Float := 1.0E-6;
   begin
      if Float(Actual) - Float(Expected) > Tolerance or else Float(Expected) - Float(Actual) > Tolerance then
         Put_Line("Test Failed: " & Test_Name);
         Put_Line("Expected: " & Expected'Image);
         Put_Line("Actual  : " & Actual'Image);
      else
         Put_Line("Test Passed: " & Test_Name);
      end if;
   end Assert_Equal;

   -- Helper procedure for gradients
   procedure Assert_Gradient(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Expected_Image : constant String := Expected.Image;
      Actual_Image   : constant String := Actual.Image;
   begin
      if Expected_Image /= Actual_Image then
         Put_Line("Test Failed: " & Test_Name);
         Put_Line("Expected Gradient: " & Expected_Image);
         Put_Line("Actual Gradient  : " & Actual_Image);
      else
         Put_Line("Test Passed: " & Test_Name);
      end if;
   end Assert_Gradient;

begin
   Put_Line("=== Cross-Entropy Loss Unit Tests ===");

   -- TC-013: Test forward pass with correct prediction
   declare
      Loss_Value : D.Element_T := Loss.Forward(Expected, Predicted_Correct);
   begin
      Assert_Equal(Expected_Loss_Correct, Loss_Value, "TC-013: Correct Prediction");
   end;

   -- TC-014: Test forward pass with incorrect prediction
   declare
      Loss_Value : D.Element_T := Loss.Forward(Expected, Predicted_Incorrect);
   begin
      Assert_Equal(Expected_Loss_Incorrect, Loss_Value, "TC-014: Incorrect Prediction");
   end;

   -- TC-015: Test backward pass gradient for correct prediction
   declare
      Expected_Gradient : D.Tensor_T := To_Tensor([-0.734732, 0.244911, 0.244911], [1, 3]);
      Gradient : D.Tensor_T := Loss.Backward(Expected, Predicted_Correct);
   begin
      Assert_Gradient(Expected_Gradient, Gradient, "TC-015: Gradient for Correct Prediction");
   end;

   -- TC-016: Test backward pass gradient for incorrect prediction
   declare
      Expected_Gradient : D.Tensor_T := To_Tensor([0.155363, -0.522944, 0.367581], [1, 3]);
      Gradient : D.Tensor_T := Loss.Backward(Expected, Predicted_Incorrect);
   begin
      Assert_Gradient(Expected_Gradient, Gradient, "TC-016: Gradient for Incorrect Prediction");
   end;

   Put_Line("=== All Cross-Entropy Loss Unit Tests Completed ===");
end Loss_Suite;
