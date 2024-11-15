with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Del;
with Del.Initializers;
with Del.Operators;

procedure Test1_Orka is
   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;

   L : DOp.Linear_T;
   X : D.Tensor_T := Ones((3, 2));
   Data : D.Tensor_T := Ones((2, 3));
   Result : D.Tensor_T := Zeros((3, 3));

begin
   -- Example usage of linear layer (if implemented)
   -- Data := DOp.Forward(L, X);

   -- Perform matrix multiplication
   Result := X * Data;
   Put_Line(Image(Result));
end Test1_Orka;
