with Del;
with Del.Operators; use Del.Operators;
with Del.Model; use Del.Model;
with Del.Loss; use Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;
with Ada.Strings;
with Ada.Float_Text_IO; use Ada.Float_Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

procedure SoftMax_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DLoss renames Del.Loss;

   -- Test dimensions
   Batch_Size : constant := 1;
   Output_Size : constant := 3;

   -- Test tensors
   Positive_Input : constant D.Tensor_T := To_Tensor([1.0, 2.0, 3.0], [Batch_Size, Output_Size]);
   Negative_Input : constant D.Tensor_T := To_Tensor([-1.0, -2.0, -3.0], [Batch_Size, Output_Size]);
   Hard_OneHot_Actual : constant D.Tensor_T := To_Tensor(
     [ -1.2515702, -1.6863402, -1.4078702,
        6.27207E+01, 1.06094E+02, 8.9826601,
        8.58803E+01, 5.86717E+01, 5.04719E+01 ], [3,3]);
   Hard_OneHot_Expected : constant D.Tensor_T := To_Tensor(
     [ 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       1.0, 0.0, 0.0 ], [3,3]);
   Soft_Actual_1 : constant D.Tensor_T := To_Tensor([0.9, 0.05, 0.05], [1,3]);
   Soft_Expected_1 : constant D.Tensor_T := To_Tensor([0.0, 1.0, 0.0], [1,3]);
   Soft_Expected_2 : constant D.Tensor_T := To_Tensor([1.0, 0.0, 0.0], [1,3]);

   -- Expected SoftMax outputs
   Expected_Positive_Output : constant D.Tensor_T := To_Tensor([0.09003057, 0.24472847, 0.66524096], [Batch_Size, Output_Size]);
   Expected_Negative_Output : constant D.Tensor_T := To_Tensor([0.66524096, 0.24472847, 0.09003057], [Batch_Size, Output_Size]);

   Test_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);
   Network_Result : D.Tensor_T := Zeros([Batch_Size, Output_Size]);

   -- Layers
   M : DOp.SoftMax_T;
   Loss : DLoss.Cross_Entropy_T;
   Network : DMod.Model;

   -- Helper procedure for tensor assertions
   procedure Assert_Test(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Tolerance : constant Float := 0.00001;
      Diff      : constant D.Tensor_T := Abs(Expected - Actual);
   begin
      if Float(Max(Diff)) > Tolerance then
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected: " & Expected.Image);
         Put_Line("Actual  : " & Actual.Image);
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Test;


begin
   Put_Line("=== SoftMax Layer and CrossEntropy Loss Tests ===");

   -- SoftMax Forward Positive Input
   Put_Line("1. Testing SoftMax Forward Positive Input");
   Test_Result := DOp.Forward(M, Positive_Input);
   Assert_Test(Expected_Positive_Output, Test_Result, "SoftMax Forward Positive Input");

   -- SoftMax Forward Negative Input
   Put_Line("2. Testing SoftMax Forward Negative Input");
   Test_Result := DOp.Forward(M, Negative_Input);
   Assert_Test(Expected_Negative_Output, Test_Result, "SoftMax Forward Negative Input");

   -- Add SoftMax to Network and run
   Put_Line("3. Testing Network with SoftMax");
   declare
      SoftMax_Layer : constant DOp.SoftMax_Access_T := new DOp.SoftMax_T;
      Temp_Network : DMod.Model;
   begin
      DMod.Add_Layer(Temp_Network, D.Func_Access_T(SoftMax_Layer));
      Network_Result := Temp_Network.Run_Layers(Positive_Input);
      Assert_Test(Expected_Positive_Output, Network_Result, "Network SoftMax Layer");
   end;


   Put_Line("=== All SoftMax + CrossEntropy Tests Completed ===");
end SoftMax_Testcases;
