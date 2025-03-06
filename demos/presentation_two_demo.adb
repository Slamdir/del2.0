with Del;
with Del.Operators;
with Del.Model;
with Del.Initializers;

with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure presentation_two_demo is

   Input : Del.Tensor_T := To_Tensor([9.0, 2.0, Del.Element_T(-4.0), Del.Element_T(-5.0), 5.0, 0.0, 3.0, 15.0, 9.0], [3,3]);

   Network : Del.Model.Model;

   Linear_Layer : Del.Operators.Linear_Access_T;
   ReLU_Layer : Del.Operators.ReLU_Access_T;
   Softmax_Layer : Del.Operators.SoftMax_Access_T;

begin

   Linear_Layer := new Del.Operators.Linear_T;
   Linear_Layer.Initialize(3, 3);
   Network.Add_Layer(Del.Func_Access_T(Linear_Layer));

   ReLU_Layer := new Del.Operators.ReLU_T;
   Network.Add_Layer(Del.Func_Access_T(ReLU_Layer));

   Softmax_Layer := new Del.Operators.SoftMax_T;
   Network.Add_Layer(Del.Func_Access_T(Softmax_Layer));

   declare 
      Result : Del.Tensor_T := Del.Model.Run_Layers(Network, Input);
   begin 
      Put_Line(Result.Image);
   end;

end presentation_two_demo;
