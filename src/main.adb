with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Main is
   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   
   -- ReLU test tensors
   Input_Data : D.Tensor_T := Ones((2, 2));
   Negative_Data : D.Tensor_T := Ones((2, 2)) * (-1.0);
   Gradient_Data : D.Tensor_T := Ones((2, 2));
   Test_Result : D.Tensor_T := Zeros((2, 2));
   
   -- Create layers and network
   R : DOp.ReLU_T;
   Network : DMod.Model;
begin
   -- First, test ReLU independently
   Put_Line("=== Independent ReLU Tests ===");
   Put_Line("1. Testing with positive values (1.0):");
   Put_Line("Input:");
   Put_Line(Image(Input_Data));
   Test_Result := R.Forward(Input_Data);
   Put_Line("After ReLU (should keep positives):");
   Put_Line(Image(Test_Result));
   
   Put_Line("Backward pass after positive input:");
   Put_Line("Gradient:");
   Put_Line(Image(Gradient_Data));
   Test_Result := R.Backward(Gradient_Data);
   Put_Line("After Backward (should pass through):");
   Put_Line(Image(Test_Result));
   
   Put_Line("2. Testing with negative values (-1.0):");
   Put_Line("Input:");
   Put_Line(Image(Negative_Data));
   Test_Result := R.Forward(Negative_Data);
   Put_Line("After ReLU (should be zeros):");
   Put_Line(Image(Test_Result));

   Put_Line("Backward pass after negative input:");
   Put_Line("Gradient:");
   Put_Line(Image(Gradient_Data));
   Test_Result := R.Backward(Gradient_Data);
   Put_Line("After Backward (should be zeros):");
   Put_Line(Image(Test_Result));

   begin
      DMod.Add_Layer(Network, new DOp.Linear_T);
      DMod.Add_Layer(Network, new DOp.ReLU_T);
   exception
      when Constraint_Error =>
         Put_Line("Error: Tensor dimensions mismatch in network. Check layer dimensions.");
      when others =>
         Put_Line("Error: Unexpected error in network execution.");
   end;
end Main;