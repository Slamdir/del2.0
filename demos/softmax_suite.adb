with Ada.Text_IO; use Ada.Text_IO;
with Del;
with Del.Operators;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Softmax_Suite is
   package D renames Del;
   package DOp renames Del.Operators;

   -- Test Inputs
   Simple_Input : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0], [1, 3]);       -- TC-009: Simple vector
   Batch_Input  : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]); -- TC-010: Batch of vectors
   Large_Input  : D.Tensor_T := To_Tensor([1000.0, 1001.0, 1002.0], [1, 3]); -- TC-011: Large numbers
   Negative_Input : D.Tensor_T := To_Tensor([-1000.0, -1001.0, -1002.0], [1, 3]); -- TC-011: Small numbers

   -- Expected outputs
   Simple_Output : D.Tensor_T := To_Tensor([0.09003057, 0.24472847, 0.66524096], [1, 3]);
   Batch_Output  : D.Tensor_T := To_Tensor([
      0.09003057, 0.24472847, 0.66524096, 
      0.09003057, 0.24472847, 0.66524096
   ], [2, 3]);
   Stable_Output : D.Tensor_T := To_Tensor([0.09003057, 0.24472847, 0.66524096], [1, 3]);

   -- Softmax Layer
   M : DOp.SoftMax_T;

   -- Helper procedure to assert equality
   procedure Assert_Equal(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Expected_Image : constant String := Expected.Image;
      Actual_Image   : constant String := Actual.Image;
   begin
      if Expected_Image /= Actual_Image then
         Put_Line("Test Failed: " & Test_Name);
         Put_Line("Expected: " & Expected_Image);
         Put_Line("Actual  : " & Actual_Image);
      else
         Put_Line("Test Passed: " & Test_Name);
      end if;
   end Assert_Equal;

begin
   Put_Line("=== Softmax Layer Unit Tests ===");

   -- TC-009: Simple vector
   Put_Line("1. Testing Simple Vector");
   Assert_Equal(Simple_Output, M.Forward(Simple_Input), "TC-009: Simple Vector");

   -- TC-010: Batch of vectors
   Put_Line("2. Testing Batch of Vectors");
   Assert_Equal(Batch_Output, M.Forward(Batch_Input), "TC-010: Batch of Vectors");

   -- TC-011: Large numbers
   Put_Line("3. Testing Numerical Stability (Large Numbers)");
   Assert_Equal(Stable_Output, M.Forward(Large_Input), "TC-011: Numerical Stability (Large Numbers)");

   -- TC-012: Negative numbers
   Put_Line("4. Testing Numerical Stability (Negative Numbers)");
   Assert_Equal(Stable_Output, M.Forward(Negative_Input), "TC-012: Numerical Stability (Negative Numbers)");

   Put_Line("=== All Softmax Layer Unit Tests Completed ===");
end Softmax_Suite;
