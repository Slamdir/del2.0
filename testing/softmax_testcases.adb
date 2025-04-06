with Del;
with Del.Operators;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure SoftMax_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test dimensions
   Batch_Size : constant := 1;
   Output_Size : constant := 3;

   -- SoftMax layer test tensors
   Positive_Input : constant D.Tensor_T := To_Tensor([1.0, 2.0, 3.0], [Batch_Size, Output_Size]);
   Negative_Input : constant D.Tensor_T := To_Tensor([-1.0, -2.0, -3.0], [Batch_Size, Output_Size]);

   -- Expected outputs
   Expected_Positive_Output : constant D.Tensor_T := To_Tensor([0.09003057, 0.24472847, 0.66524096], [Batch_Size, Output_Size]);
   Expected_Negative_Output : constant D.Tensor_T := To_Tensor([0.66524096, 0.24472847, 0.09003057], [Batch_Size, Output_Size]);

   Test_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]); -- Initialize with zeros

   -- Create layer and network
   M : DOp.SoftMax_T;
   Network : DMod.Model;

   -- Helper procedure to assert test outcomes
   procedure Assert_Test(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Tolerance : constant Float_32 := Float_32(0.00001);
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
   Put_Line("=== SoftMax Layer Tests ===");

   -- Test Forward Pass with Positive Input
   Put_Line("1. Testing Forward Pass with Positive Input");
   Test_Result := M.Forward(Positive_Input);
   Assert_Test(Expected_Positive_Output, Test_Result, "SoftMax Forward Positive Input");

   -- Test Forward Pass with Negative Input
   Put_Line("2. Testing Forward Pass with Negative Input");
   Test_Result := M.Forward(Negative_Input);
   Assert_Test(Expected_Negative_Output, Test_Result, "SoftMax Forward Negative Input");

   -- Add SoftMax Layer to Network
   Put_Line("3. Adding SoftMax Layer to Network");
   declare
      SoftMax_Layer : DOp.SoftMax_Access_T := new DOp.SoftMax_T;
      Network_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   begin
      DMod.Add_Layer(Network, D.Func_Access_T(SoftMax_Layer));

      -- Run the network forward
      Network_Result := Network.Run_Layers(Positive_Input);
      Assert_Test(Expected_Positive_Output, Network_Result, "Network SoftMax Layer Test");
   end;

   Put_Line("=== All SoftMax Layer Tests Completed ===");
end SoftMax_Testcases;