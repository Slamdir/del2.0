with Del;
with Del.Operators;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Doubles.Tensors.CPU; use Orka.Numerics.Doubles.Tensors.CPU;
with Orka.Numerics.Doubles.Tensors; use Orka.Numerics.Doubles.Tensors;
with Orka; use Orka;

procedure Linear_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test dimensions
   Input_Size  : constant := 2;
   Output_Size : constant := 3;
   Batch_Size  : constant := 2;

   -- Linear layer test tensors
   Positive_Input : D.Tensor_T := Ones([Batch_Size, Input_Size]);          -- Positive 2x2 input
   Negative_Input : D.Tensor_T := Ones([Batch_Size, Input_Size]) * D.Element_T(-1.0); -- Negative 2x2 input
   Gradient_Data  : D.Tensor_T := Ones([Batch_Size, Output_Size]);         -- 2x3 gradient

   -- Create layer and network
   L : DOp.Linear_T;
   Network : DMod.Model;

   -- Helper procedure to assert tensor closeness
   procedure Assert_Tensor_Close(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Tolerance : constant Float_32 := Float_32(0.0001);
      Diff      : constant D.Tensor_T := Abs(Expected - Actual);
   begin
      if Float_32(Max(Diff)) > Tolerance then
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected: " & Expected.Image);
         Put_Line("Actual  : " & Actual.Image);
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Tensor_Close;

begin
   Put_Line("=== Linear Layer Full Testcases ===");

   -- Initialize the layer
   L.Initialize(Input_Size, Output_Size);

   -- 1. Test Forward Pass with Positive Input
   Put_Line("1. Testing Forward Pass with Positive Input");
   declare
      Forward_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   begin
      Forward_Result := L.Forward(Positive_Input);
      Put_Line("Forward Positive Input Result:");
      Put_Line(Forward_Result.Image);
      -- No hardcoded expected because weights are random, only shape match checked.
      if Shape(Forward_Result) = [Batch_Size, Output_Size] then
         Put_Line("Forward Pass Positive Input Test Passed");
      else
         Put_Line("Forward Pass Positive Input Test Failed");
      end if;
   end;

   -- 2. Test Forward Pass with Negative Input
   Put_Line("2. Testing Forward Pass with Negative Input");
   declare
      Forward_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   begin
      Forward_Result := L.Forward(Negative_Input);
      Put_Line("Forward Negative Input Result:");
      Put_Line(Forward_Result.Image);
      if Shape(Forward_Result) = [Batch_Size, Output_Size] then
         Put_Line("Forward Pass Negative Input Test Passed");
      else
         Put_Line("Forward Pass Negative Input Test Failed");
      end if;
   end;

   -- 3. Test Backward Pass
   Put_Line("3. Testing Backward Pass");
   declare
      Backward_Result : D.Tensor_T := Zeros([Batch_Size, Input_Size]);
   begin
      Backward_Result := L.Backward(Gradient_Data);
      Put_Line("Backward Result (dL/dX):");
      Put_Line(Backward_Result.Image);
      if Shape(Backward_Result) = [Batch_Size, Input_Size] then
         Put_Line("Backward Pass Shape Test Passed");
      else
         Put_Line("Backward Pass Shape Test Failed");
      end if;
   end;

   -- 4. Testing inside a Network
   Put_Line("4. Testing in Network Context");
   declare
      Linear_Layer : constant DOp.Linear_Access_T := new DOp.Linear_T;
      Network_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   begin
      Linear_Layer.Initialize(Input_Size, Output_Size);
      DMod.Add_Layer(Network, D.Func_Access_T(Linear_Layer));

      Network_Result := Network.Run_Layers(Positive_Input);
      Put_Line("Network Forward Pass Result:");
      Put_Line(Network_Result.Image);
      if Shape(Network_Result) = [Batch_Size, Output_Size] then
         Put_Line("Network Layer Forward Test Passed");
      else
         Put_Line("Network Layer Forward Test Failed");
      end if;
   end;

   Put_Line("=== All Linear Layer Tests Completed ===");
end Linear_Testcases;
