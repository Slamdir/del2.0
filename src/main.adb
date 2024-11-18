with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Main is

   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   --Data_1 : D.Tensor_T := Zeros((2, 2));
   X : D.Tensor_T := Ones((3, 2));
   Data : D.Tensor_T := Ones((2, 3));

   Result : D.Tensor_T := Zeros((3, 3));

   L : DOp.Linear_T;
   R : DOp.ReLU_T;

   Network : DMod.Model;

begin
   --L.D.Insert ("Data", Data_1);
   --Put_Line (Image (L.D.Element ("Data")));

   --Put_Line(Image(Data));
   --Data := DOp.Forward(L, X);
   --Put_Line (Image(Data));

   --  Result := X * Data;
   --  Put_Line(Image(Result));

   DMod.Add_Layer(Network, new DOp.Linear_T);
   DMod.Add_Layer(Network, new DOp.ReLU_T);

   DMod.Run_Layers(Network);

end Main;
