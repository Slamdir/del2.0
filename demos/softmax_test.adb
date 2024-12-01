with Del;
with Del.Operators;
with Del.Initializers;
with Del.Model;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Softmax_Test is

   package D renames Del;
   package DI renames Del.Initializers;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DLoss renames Del.Loss;

   --Data_1 : D.Tensor_T := Zeros((2, 2));
   X : D.Tensor_T := To_Tensor ([5.0, 2.0, 3.0, D.Element_T(-5.0), 5.0, 6.0, 3.0, 15.0, 9.0], [3,3]);
   Data : D.Tensor_T := Ones((2, 3));

   Result : D.Tensor_T := Zeros((3, 3));

   L : DOp.Linear_T;
   R : DOp.ReLU_T;
   M : DOp.SoftMax_T;

   Network : DMod.Model;
   
   --  Output : constant D.Tensor_T := Sum(X, 1);
   Output2 : constant D.Tensor_T := DOp.Forward(M, X);
   --  Output3 : D.Tensor_T := Log10(X);

   Expected : D.Tensor_T := To_Tensor ([1.0, 0.0], [1, 2]);
   Actual   : D.Tensor_T := To_Tensor ([2.0, 1.0], [1, 2]);
   Loss     : DLoss.Cross_Entropy_T;

begin
   --L.D.Insert ("Data", Data_1);
   --Put_Line (Image (L.D.Element ("Data")));

   --Put_Line(Image(Data));
   --Data := DOp.Forward(L, X);
   --Put_Line (Image(Data));

   --  Result := X * Data;
   --  Put_Line(Image(Result));

   --  DMod.Add_Layer(Network, new DOp.Linear_T);
   --  DMod.Add_Layer(Network, new DOp.ReLU_T);

   --  DMod.Run_Layers(Network);

   --  Put_Line(Image(Output2));
   --  Put_Line(Image(Output3));

   Put_Line(Loss.Forward(Expected, Actual)'Image);
   --  Put_Line(Loss.Backward(Expected, Actual).Image);

end Softmax_Test;