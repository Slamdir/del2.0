with Del;
with Del.Operators;
with Del.Model;
with Del.Initializers;
with Del.Optimizers;

with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Optim_Test is

    Optim : Del.Optimizers.SGD_T := Del.Optimizers.Create_SGD_T(Learning_Rate => 0.0001, Weight_Decay => 0.05, Momentum => 0.001);
    --  Optim2 : Del.Optimizers.SGD_T;

   Network : Del.Model.Model;
   Linear_Layer : Del.Operators.Linear_Access_T;
   ReLU_Layer : Del.Operators.ReLU_Access_T;

begin
   
   Linear_Layer := new Del.Operators.Linear_T;
   Linear_Layer.Initialize(3, 3);
   Network.Add_Layer(Del.Func_Access_T(Linear_Layer));

   ReLU_Layer := new Del.Operators.ReLU_T;
   Network.Add_Layer(Del.Func_Access_T(ReLU_Layer));

   Linear_Layer := new Del.Operators.Linear_T;
   Linear_Layer.Initialize(3, 5);
   Network.Add_Layer(Del.Func_Access_T(Linear_Layer));

   --  Put_Line("First Step");
   --  Optim.Step(Network.Get_Params);
   
   --  Put_Line("Second Step");
   --  Optim.Step(Network.Get_Params);

end Optim_Test;