with Del;
with Del.Operators;
with Del.Model;
with Del.Optimizers;

with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Doubles.Tensors.CPU; use Orka.Numerics.Doubles.Tensors.CPU;
with Orka; use Orka;

procedure Optimizer_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DOpt renames Del.Optimizers;

   Optim : DOpt.SGD_T := DOpt.Create_SGD_T(Learning_Rate => 0.0001, Weight_Decay => 0.05, Momentum => 0.001);

   Network : DMod.Model;
   Linear_Layer : DOp.Linear_Access_T;
   ReLU_Layer   : DOp.ReLU_Access_T;

   -- Assertion
   procedure Assert_Parameters_Changed(Before, After : D.Tensor_T; Test_Name : String) is
      Diff : constant D.Tensor_T := Abs(Before - After);
   begin
      if Float_32(Max(Diff)) > 0.0 then
         Put_Line(Test_Name & " Passed (Parameters updated)");
      else
         Put_Line(Test_Name & " Failed (Parameters did not change)");
         Put_Line("Before:");
         Put_Line(Before.Image);
         Put_Line("After:");
         Put_Line(After.Image);
      end if;
   end Assert_Parameters_Changed;

begin
   Put_Line("=== Optimizer Simple Testcases ===");

   -- Build the network
   Linear_Layer := new DOp.Linear_T;
   Linear_Layer.Initialize(3, 3);
   Network.Add_Layer(D.Func_Access_T(Linear_Layer));

   ReLU_Layer := new DOp.ReLU_T;
   Network.Add_Layer(D.Func_Access_T(ReLU_Layer));

   Linear_Layer := new DOp.Linear_T;
   Linear_Layer.Initialize(3, 5);
   Network.Add_Layer(D.Func_Access_T(Linear_Layer));

   -- Capture before step
   declare
      --Before_Step : D.Tensor_T := Layers(1).all.As_Linear.Get_Params(0).all;
   begin
      -- Step
      Put_Line("First Optimizer Step");
      Optim.Step(Network.Get_Layers_Vector);

      -- Capture after step
      declare
         After_Step : D.Tensor_T := Layers(1).all.As_Linear.Get_Params(0).all;
      begin
         Assert_Parameters_Changed(Before_Step, After_Step, "Optimizer Step Test");
      end;
   end;

   Put_Line("=== Optimizer Simple Testcases Completed ===");
end Optimizer_Testcases;
