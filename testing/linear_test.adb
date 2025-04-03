with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure linear_test is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test dimensions
   Input_Size  : constant := 2;
   Output_Size : constant := 3;
   Batch_Size  : constant := 2;

   -- Linear layer test tensors
   Positive_Input : D.Tensor_T := Ones([Batch_Size, Input_Size]);          -- Positive 2x2 input
   Negative_Input : D.Tensor_T := Ones([Batch_Size, Input_Size]) * Float_32(-1.0); -- Negative 2x2 input
   Gradient_Data  : D.Tensor_T := Ones([Batch_Size, Output_Size]);         -- 2x3 gradient

   -- Updated expected outputs
   Expected_Positive_Output : D.Tensor_T := To_Tensor(
      [Float_32(-0.2), Float_32(-0.2), Float_32(-0.2), Float_32(-0.2), Float_32(-0.2), Float_32(-0.2)], 
      [Batch_Size, Output_Size]
   );
   Expected_Negative_Output : D.Tensor_T := To_Tensor(
      [Float_32(0.2), Float_32(0.2), Float_32(0.2), Float_32(0.2), Float_32(0.2), Float_32(0.2)], 
      [Batch_Size, Output_Size]
   );
   Expected_Backward_Output : D.Tensor_T := To_Tensor(
      [Float_32(-0.1), Float_32(-0.1), Float_32(-0.1), Float_32(-0.1), Float_32(-0.1), Float_32(-0.1)], 
      [Batch_Size, Output_Size]
   );

   Test_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);           -- Initialize with zeros

   -- Create layer and network
   L : DOp.Linear_T;
   Network : DMod.Model;

   -- Helper procedure to assert test outcomes
   procedure Assert_Test(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Tolerance : constant Float_32 := Float_32(0.0001);
      Diff      : D.Tensor_T := Abs(Expected - Actual);
   begin
      if Max(Diff) > Tolerance then
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected: " & Expected.Image);
         Put_Line("Actual  : " & Actual.Image);
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Test;

begin
   Put_Line("=== Linear Layer Tests ===");

   -- Initialize the layer
   L.Initialize(Input_Size, Output_Size);

   -- Debug weights, biases, and input
   declare
      Params : D.Params_T := L.Get_Params;
      Weights : D.Tensor_T := Params(0).all;
      Bias : D.Tensor_T := Params(1).all;
   begin
      Put_Line("Initialized Weights:");
      Put_Line(Weights.Image);
      Put_Line("Initialized Bias:");
      Put_Line(Bias.Image);
      Put_Line("Positive Input:");
      Put_Line(Positive_Input.Image);
   end;

   -- Test Forward Pass with Positive Input
   Put_Line("1. Testing Forward Pass with Positive Input");
   Test_Result := L.Forward(Positive_Input);
   Assert_Test(Expected_Positive_Output, Test_Result, "Forward Pass Positive Input");

   -- Test Forward Pass with Negative Input
   Put_Line("2. Testing Forward Pass with Negative Input");
   Test_Result := L.Forward(Negative_Input);
   Assert_Test(Expected_Negative_Output, Test_Result, "Forward Pass Negative Input");

   -- Add Layer to Network
   Put_Line("3. Adding Layer to Network");
   declare
      Linear_Layer : DOp.Linear_Access_T := new DOp.Linear_T;
      Network_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   begin
      Linear_Layer.Initialize(Input_Size, Output_Size);
      DMod.Add_Layer(Network, D.Func_Access_T(Linear_Layer));

      -- Run the network forward
      Network_Result := Network.Run_Layers(Positive_Input);
      Assert_Test(Expected_Positive_Output, Network_Result, "Network Layer Test");
   end;

   Put_Line("=== All Linear Layer Tests Completed ===");
end linear_test;
