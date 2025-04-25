with Del;
with Del.Operators;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure ReLU_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test dimensions
   Batch_Size  : constant := 2;
   Input_Size  : constant := 2;

   -- ReLU test tensors
   Positive_Input : D.Tensor_T := To_Tensor([1.0, 1.0, 1.0, 1.0], [Batch_Size, Input_Size]);
   Negative_Input : D.Tensor_T := To_Tensor([-1.0, -1.0, -1.0, -1.0], [Batch_Size, Input_Size]);
   Mixed_Input    : D.Tensor_T := To_Tensor([1.0, -1.0, 0.0, 2.0], [Batch_Size, Input_Size]);

   Expected_Positive_Output : D.Tensor_T := To_Tensor([1.0, 1.0, 1.0, 1.0], [Batch_Size, Input_Size]);
   Expected_Negative_Output : D.Tensor_T := To_Tensor([0.0, 0.0, 0.0, 0.0], [Batch_Size, Input_Size]);
   Expected_Mixed_Output    : D.Tensor_T := To_Tensor([1.0, 0.0, 0.0, 2.0], [Batch_Size, Input_Size]);

   Gradient_Input : D.Tensor_T := To_Tensor([1.0, 1.0, 1.0, 1.0], [Batch_Size, Input_Size]);
   Expected_Mixed_Backward : D.Tensor_T := To_Tensor([1.0, 0.0, 0.0, 1.0], [Batch_Size, Input_Size]);

   Test_Result : D.Tensor_T := Zeros([Batch_Size, Input_Size]);

   -- Create ReLU layer
   R : DOp.ReLU_T;
   Network : DMod.Model;

   -- Helper procedure for assertions
   procedure Assert_Test(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Tolerance : constant Float_32 := 0.0001;
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
   Put_Line("=== ReLU Layer Tests ===");

   -- Test Forward Pass with Positive Input
   Put_Line("1. Testing Forward Pass with Positive Input");
   Test_Result := DOp.Forward(R, Positive_Input);
   Assert_Test(Expected_Positive_Output, Test_Result, "ReLU Forward Positive Input");

   -- Test Forward Pass with Negative Input
   Put_Line("2. Testing Forward Pass with Negative Input");
   Test_Result := DOp.Forward(R, Negative_Input);
   Assert_Test(Expected_Negative_Output, Test_Result, "ReLU Forward Negative Input");

   -- Test Forward Pass with Mixed Input
   Put_Line("3. Testing Forward Pass with Mixed Input");
   Test_Result := DOp.Forward(R, Mixed_Input);
   Assert_Test(Expected_Mixed_Output, Test_Result, "ReLU Forward Mixed Input");

   -- Test Backward Pass with Mixed Input
   Put_Line("4. Testing Backward Pass with Mixed Input");
   declare
      Ignore : D.Tensor_T := DOp.Forward(R, Mixed_Input); -- capture mask internally
   begin
      Test_Result := DOp.Backward(R, Gradient_Input);
      Assert_Test(Expected_Mixed_Backward, Test_Result, "ReLU Backward Mixed Input");
   end;

   -- Add ReLU Layer to Network and test execution
   Put_Line("5. Adding ReLU Layer to Network and running network");
   declare
      ReLU_Layer : DOp.ReLU_Access_T := new DOp.ReLU_T;
      Network_Result : D.Tensor_T := Zeros([Batch_Size, Input_Size]);
   begin
      DMod.Add_Layer(Network, D.Func_Access_T(ReLU_Layer));
      Network_Result := Network.Run_Layers(Mixed_Input);
      Assert_Test(Expected_Mixed_Output, Network_Result, "Network ReLU Layer Test");
   end;

   Put_Line("6. Adding multiple ReLU layers and testing");
   declare
      L1 : DOp.ReLU_Access_T := new DOp.ReLU_T;
      L2 : DOp.ReLU_Access_T := new DOp.ReLU_T;
      Model : DMod.Model;
      Result : D.Tensor_T := Zeros([Batch_Size, Input_Size]);
   begin
      DMod.Add_Layer(Model, D.Func_Access_T(L1));
      DMod.Add_Layer(Model, D.Func_Access_T(L2));
      Result := Model.Run_Layers(Mixed_Input);
      Assert_Test(Expected_Mixed_Output, Result, "Network Double ReLU Layer Test");
   end;

   Put_Line("=== All ReLU Layer Tests Completed ===");
end ReLU_Testcases;
