
with Del; 
with Del.Operators;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Main is

   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;

   Data_1 : D.Tensor_T := Zeros((2, 2));
   Data_2 : D.Tensor_T := Zeros((4, 4));

   L : DOp.Linear_T;

begin
   L.D.Insert ("Data", Data_1);
   L.D.Insert ("Grad", Data_2);

   Put_Line (Image (L.D.Element ("Data")));
   Put_Line (Image (L.D.Element ("Grad")));

   Put_Line (L.D'Image);

end Main;
