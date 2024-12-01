with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Main_Demo is

   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   X : D.Tensor_T := To_Tensor ([5.0, 2.0, 3.0, D.Element_T(-5.0), 5.0, 6.0, 3.0, 15.0, 9.0], [3,3]);

   Network : DMod.Model;

   Linear_Layer : DOp.Linear_Access_T;
   ReLU_Layer : DOp.ReLU_Access_T;
   Soft_Layer : DOp.SoftMax_Access_T;
   Holder : D.Tensor_T := Zeros((3, 3));

begin

   Linear_Layer := new DOp.Linear_T;
   --  Network.Add_Layer(D.Func_Access_T(Linear_Layer));

   ReLU_Layer := new DOp.ReLU_T;
   Network.Add_Layer(D.Func_Access_T(ReLU_Layer));

   Soft_Layer := new DOp.SoftMax_T;
   Network.Add_Layer(D.Func_Access_T(Soft_Layer));
   
   Holder := DMod.Run_Layers(Network, X);
   Put_Line(Holder.Image);

end Main_Demo;
