with Del;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure MSE_test is

   package D renames Del;
   package DLoss renames Del.Loss;

   procedure TestForward1 (Loss : DLoss.Mean_Square_Error_T; Desired_Value : Float) is

   Actual   : D.Tensor_T := To_Tensor (
      [10.0, 20.0, 30.0, 40.0, 50.0], [1, 5]
   );

   Expected   : D.Tensor_T := To_Tensor (
      [12.0, 18.0, 32.0, 38.0, 48.0], [1, 5]
   );

   begin
      Put_Line("Testing with: ");
      Put_Line("Actual:    " & Actual.Image);
      Put_Line("Expected:  " & Expected.Image);
      New_Line;
      Put_Line("Loss Got: " & Loss.Forward(Expected, Actual)'Image);
      Put_Line("Desired Is: " & Desired_Value'Image);
   end TestForward1;

   procedure TestForward2 (Loss : DLoss.Mean_Square_Error_T; Desired_Value : Float) is

   Actual   : D.Tensor_T := To_Tensor (
      [1.0, 2.0, 3.0, 4.0], [2, 2]
   );

   Expected   : D.Tensor_T := To_Tensor (
      [1.5, 2.5, 2.5, 3.5], [2, 2]
   );

   begin
      Put_Line("Testing with: ");
      Put_Line("Actual:    " & Actual.Image);
      Put_Line("Expected:  " & Expected.Image);
      New_Line;
      Put_Line("Loss Got: " & Loss.Forward(Expected, Actual)'Image);
      Put_Line("Desired Is: " & Desired_Value'Image);
   end TestForward2;

   Loss  : DLoss.Mean_Square_Error_T;

begin

   TestForward1 (Loss, 4.0);
   New_Line;
   TestForward2 (Loss, 0.25);

end MSE_test;