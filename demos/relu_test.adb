with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure ReLU_Test is
   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- ReLU test tensors
   Input_Data : D.Tensor_T := Ones((2, 2));
   Negative_Data : D.Tensor_T := Ones((2, 2)) * (-1.0);
   Gradient_Data : D.Tensor_T := Ones((2, 2));
   Test_Result : D.Tensor_T := Zeros((2, 2));
   Expected_Positive : D.Tensor_T := Ones((2, 2));
   Expected_Negative : D.Tensor_T := Zeros((2, 2));

   -- Create ReLU layer
   R : DOp.ReLU_T;

   -- Helper procedure for assertion
   procedure Assert_Equal(Expected, Actual : D.Tensor_T; Test_Name : String) is
   begin
      if Expected.Image = Actual.Image then
         Put_Line(Test_Name & " Passed");
      else
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected:");
         Put_Line(Expected.Image);
         Put_Line("Actual:");
         Put_Line(Actual.Image);
      end if;
   end Assert_Equal;

begin
   -- Independent ReLU Tests
   Put_Line("=== Independent ReLU Tests ===");

   -- Test 1: Forward pass with positive input
   Put_Line("1. Testing ReLU Forward with positive values (1.0):");
   Test_Result := R.Forward(Input_Data);
   Assert_Equal(Expected_Positive, Test_Result, "ReLU Forward Positive Test");

   -- Test 3: Forward pass with negative input
   Put_Line("2. Testing ReLU Forward with negative values (-1.0):");
   Test_Result := R.Forward(Negative_Data);
   Assert_Equal(Expected_Negative, Test_Result, "ReLU Forward Negative Test");

   declare
      Network : DMod.Model;
   begin
      -- Add ReLU layer to the network and ensure no exceptions occur
      Put_Line("3. Testing ReLU Layer in Network:");
      DMod.Add_Layer(Network, new DOp.ReLU_T);
      Put_Line("ReLU Layer added successfully to the network. Test Passed.");
   exception
      when Constraint_Error =>
         Put_Line("Error: Tensor dimensions mismatch in network. Test Failed.");
      when others =>
         Put_Line("Error: Unexpected error in network execution. Test Failed.");
   end;

   Put_Line("=== ReLU Tests Completed ===");
end ReLU_Test;
