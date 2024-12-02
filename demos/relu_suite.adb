with Ada.Text_IO; use Ada.Text_IO;
with Del;
with Del.Operators;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka; -- Added to make Float_32 operators visible

procedure ReLU_Suite is
   -- Alias for convenience
   package D renames Del;
   package DOp renames Del.Operators;

   -- Test inputs
   Positive_Input : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0], [3]);
   Negative_Input : D.Tensor_T := To_Tensor([-1.0, -2.0, -3.0], [3]);
   Mixed_Input    : D.Tensor_T := To_Tensor([-1.0, 0.0, 1.0], [3]);
   Shape_Input    : D.Tensor_T := To_Tensor([1.0, -1.0, 0.0, 2.0], [2, 2]);

   -- Expected outputs
   Positive_Output : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0], [3]);
   Negative_Output : D.Tensor_T := To_Tensor([0.0, 0.0, 0.0], [3]);
   Mixed_Output    : D.Tensor_T := To_Tensor([0.0, 0.0, 1.0], [3]);
   Shape_Output    : D.Tensor_T := To_Tensor([1.0, 0.0, 0.0, 2.0], [2, 2]);

   -- ReLU Layer
   ReLU_Layer : DOp.ReLU_T;

   -- Helper procedure to assert equality
   procedure Assert_Equal(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Expected_Image : String := Expected.Image;
      Actual_Image   : String := Actual.Image;
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
   -- TC-001: Test ReLU forward pass with positive values
   Assert_Equal(Positive_Output, ReLU_Layer.Forward(Positive_Input), "TC-001: Positive Values Test");

   -- TC-002: Test ReLU forward pass with negative values
   Assert_Equal(Negative_Output, ReLU_Layer.Forward(Negative_Input), "TC-002: Negative Values Test");

   -- TC-003: Verify output shape matches input shape for a 2D tensor
   Assert_Equal(Shape_Output, ReLU_Layer.Forward(Shape_Input), "TC-003: Shape Preservation Test");

   -- TC-004: Test ReLU backward pass gradient for positive values
   declare
      Grad_Input    : D.Tensor_T := To_Tensor([1.0, 1.0, 1.0], [3]);
      Expected_Grad : D.Tensor_T := To_Tensor([1.0, 1.0, 1.0], [3]);
   begin
      Assert_Equal(Expected_Grad, ReLU_Layer.Backward(Grad_Input), "TC-004: Gradient Test for Positive Values");
   end;

   Put_Line("All ReLU Tests Completed.");
end ReLU_Suite;
