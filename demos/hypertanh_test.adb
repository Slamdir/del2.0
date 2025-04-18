with Del;
with Del.Operators;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure hypertanh_test is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   
   -- HyperTanh test tensors
   Input_Data : D.Tensor_T := Ones((2, 2));
   Negative_Data : D.Tensor_T := Ones((2, 2)) * (-1.0);
   Gradient_Data : D.Tensor_T := Ones((2, 2));
   Test_Result : D.Tensor_T := Zeros((2, 2));
   
   -- Create layers and network
   H : DOp.HyperTanh_T;
   Network : DMod.Model;
begin
   Put_Line("=== Independent HyperTanh Tests ===");
   Put_Line("1. Testing with positive values (1.0):");
   Put_Line("Input:");
   Put_Line(Image(Input_Data));
   Test_Result := H.Forward(Input_Data);
   Put_Line("Expected Output (manually computed):");
   Put_Line("[[0.76159416, 0.76159416],");
   Put_Line(" [0.76159416, 0.76159416]]");
   Put_Line("After HyperTanh:");
   Put_Line(Image(Test_Result));
   Put_Line("");
   Put_Line("");
   Put_Line("");
   
   Put_Line("Backward pass after positive input:");
   Put_Line("Gradient:");
   Put_Line(Image(Gradient_Data));
   Test_Result := H.Backward(Gradient_Data);
   Put_Line("Expected Output (manually computed):");
   Put_Line("[[0.41997434, 0.41997434],");
   Put_Line(" [0.41997434, 0.41997434]]");
   Put_Line("After Backward:");
   Put_Line(Image(Test_Result));
   Put_Line("");
   Put_Line("");
   Put_Line("");
   
   Put_Line("2. Testing with negative values (-1.0):");
   Put_Line("Input:");
   Put_Line(Image(Negative_Data));
   Test_Result := H.Forward(Negative_Data);
   Put_Line("Expected Output (manually computed):");
   Put_Line("[[-0.76159416, -0.76159416],");
   Put_Line(" [-0.76159416, -0.76159416]]");
   Put_Line("After HyperTanh:");
   Put_Line(Image(Test_Result));
   Put_Line("");
   Put_Line("");
   Put_Line("");

   Put_Line("Backward pass after negative input:");
   Put_Line("Gradient:");
   Put_Line(Image(Gradient_Data));
   Test_Result := H.Backward(Gradient_Data);
   Put_Line("Expected Output (manually computed):");
   Put_Line("[[0.41997434, 0.41997434],");
   Put_Line(" [0.41997434, 0.41997434]]");
   Put_Line("After Backward:");
   Put_Line(Image(Test_Result));

   begin
      DMod.Add_Layer(Network, new DOp.HyperTanh_T);
   exception
      when Constraint_Error =>
         Put_Line("Error: Tensor dimensions mismatch in network. Check layer dimensions.");
      when others =>
         Put_Line("Error: Unexpected error in network execution.");
   end;
end hypertanh_test;