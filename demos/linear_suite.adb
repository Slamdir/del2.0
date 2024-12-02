with Ada.Text_IO; use Ada.Text_IO;
with Del;
with Del.Operators;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Linear_Suite is
   package D renames Del;
   package DOp renames Del.Operators;

   -- Test dimensions
   Input_Size  : constant := 2;
   Output_Size : constant := 2;
   Batch_Size  : constant := 2;

   -- Test Inputs
   Single_Input  : D.Tensor_T := To_Tensor([1.0], [1, 1]);       -- TC-005: Single value input
   Small_Input   : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);  -- TC-006: 2x2 input
   Batch_Input   : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);  -- TC-007: Batch input
   Large_Input   : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]); -- TC-008: Large input

   -- Expected outputs for TC-005 (Manually calculated)
   Single_Weights : D.Tensor_T := To_Tensor([0.5], [1, 1]);
   Single_Bias    : D.Tensor_T := To_Tensor([0.1], [1, 1]);
   Single_Output  : D.Tensor_T := To_Tensor([0.6], [1, 1]);

   -- Create layer
   L : DOp.Linear_T;

   -- Helper procedure to set weights and biases
   procedure Set_Params(Layer : in out DOp.Linear_T; W, B : D.Tensor_T) is
      Params : D.Params_T := Layer.Get_Params;
   begin
      -- Overwrite weights and biases
      Params(0).all := W;
      Params(1).all := B;

      -- Debug print for verification
      Put_Line("Set Weights:");
      Put_Line(W.Image);
      Put_Line("Set Bias:");
      Put_Line(B.Image);
   end Set_Params;

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
   Put_Line("=== Linear Layer Unit Tests ===");

   -- TC-005: Single value input
   Put_Line("1. Testing Single Value Input");
   L.Initialize(1, 1);
   Set_Params(L, Single_Weights, Single_Bias);
   Assert_Equal(Single_Output, L.Forward(Single_Input), "TC-005: Single Value Input");

   -- TC-006: Small matrices
   Put_Line("2. Testing Small Matrices");
   L.Initialize(Input_Size, Output_Size);
   Set_Params(
      L,
      To_Tensor([0.5, -0.2, 0.3, 0.8], [2, 2]),
      To_Tensor([0.1, -0.1], [1, 2])
   );
   Assert_Equal(
      To_Tensor([1.2, 1.3, 2.3, 2.5], [2, 2]),
      L.Forward(Small_Input),
      "TC-006: Small Matrices"
   );

   -- TC-007: Batch input
   Put_Line("3. Testing Batch Input");
   L.Initialize(Input_Size, Output_Size);
   Set_Params(
      L,
      To_Tensor([0.5, -0.1, 0.2, 0.8], [2, 2]),
      To_Tensor([0.1, 0.2], [1, 2])
   );
   Assert_Equal(
      To_Tensor([1.2, 2.3, 2.2, 4.3], [2, 2]),
      L.Forward(Batch_Input),
      "TC-007: Batch Input"
   );

   -- TC-008: Large input
   Put_Line("4. Testing Large Input");
   L.Initialize(Input_Size, Output_Size);
   Set_Params(
      L,
      To_Tensor([0.4, -0.2, 0.3, 0.7], [2, 2]),
      To_Tensor([0.0, 0.1], [1, 2])
   );
   Assert_Equal(
      To_Tensor([0.6, 1.8, 1.2, 3.4, 1.8, 5.0], [3, 2]),
      L.Forward(Large_Input),
      "TC-008: Large Input"
   );

   Put_Line("=== All Linear Layer Unit Tests Completed ===");
end Linear_Suite;
