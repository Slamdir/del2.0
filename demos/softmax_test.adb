with Del;
with Del.Operators;
with Del.Model;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Softmax_Test is

   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DLoss renames Del.Loss;

   --Put_Line(Loss.Forward(Expected, Actual)'Image);
   --Put_Line(Loss.Backward(Expected, Actual).Image);

   procedure TestForward1 (Loss : DLoss.Cross_Entropy_T) is
      Actual   : D.Tensor_T := To_Tensor (
        [Del.Element_T(-1.2515702), Del.Element_T(-1.6863402), Del.Element_T(-1.4078702),
         6.27207E+01, 1.06094E+02, 8.9826601,
         8.58803E+01, 5.86717E+01, 5.04719E1], [3, 3]);

      Expected : D.Tensor_T := To_Tensor (
        [ 1.00000E+00, 0.00000E+00, 0.00000E+00,
        0.00000E+00, 1.00000E+00, 0.00000E+00,
        1.00000E+00, 0.00000E+00, 0.00000E+00], [3, 3]);
   begin
      Put_Line("Testing with: ");
      Put_Line("Actual:    " & Actual.Image);
      Put_Line("Expected:  " & Expected.Image);
      New_Line;
      Put_Line(Loss.Forward(Expected, Actual)'Image);
   end TestForward1;

   procedure TestForward2 (Loss : DLoss.Cross_Entropy_T) is
      Actual   : D.Tensor_T := To_Tensor ([0.9, 0.05, 0.05], [1, 3]);
      Expected : D.Tensor_T := To_Tensor ([0.0, 1.0, 0.0], [1, 3]);
   begin
      Put_Line("Testing with: ");
      Put_Line("Actual:    " & Actual.Image);
      Put_Line("Expected:  " & Expected.Image);
      New_Line;
      Put_Line(Loss.Forward(Expected, Actual)'Image);
   end TestForward2;

   procedure TestForward3 (Loss : DLoss.Cross_Entropy_T) is
      Actual   : D.Tensor_T := To_Tensor ([0.9, 0.05, 0.05], [1, 3]);
      Expected : D.Tensor_T := To_Tensor ([1.0, 0.0, 0.0], [1, 3]);
   begin
      Put_Line("Testing with: ");
      Put_Line("Actual:    " & Actual.Image);
      Put_Line("Expected:  " & Expected.Image);
      New_Line;
      Put_Line(Loss.Forward(Expected, Actual)'Image);
   end TestForward3;

   Loss     : DLoss.Cross_Entropy_T;

begin

   --  TestForward1(Loss);
   TestForward2(Loss);
   TestForward3(Loss);

end Softmax_Test;