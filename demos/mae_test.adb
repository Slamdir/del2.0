with Del;
with Del.Operators;
with Del.Model;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure MAE_Test is

   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DLoss renames Del.Loss;

   procedure TestForward1 (Loss : DLoss.Mean_Absolute_Error_T; Desired_Value : Float) is

   Actual   : constant D.Tensor_T := To_Tensor (
      [10.0, 20.0, 30.0, 40.0, 50.0], [1, 5]
   );

   Expected   : constant D.Tensor_T := To_Tensor (
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

   procedure TestForward2 (Loss : DLoss.Mean_Absolute_Error_T; Desired_Value : Float) is

   Actual   : constant D.Tensor_T := To_Tensor (
      [1.0, 2.0, 3.0, 4.0], [2, 2]
   );

   Expected   : constant D.Tensor_T := To_Tensor (
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

   Loss : DLoss.Mean_Absolute_Error_T;

begin
   -- Test case 1: |12-10| + |18-20| + |32-30| + |38-40| + |48-50| = 2 + 2 + 2 + 2 + 2 = 10
   -- Average = 10/5 = 2.0
   TestForward1 (Loss, 2.0);
   New_Line;
   
   -- Test case 2: |1.5-1| + |2.5-2| + |2.5-3| + |3.5-4| = 0.5 + 0.5 + 0.5 + 0.5 = 2.0
   -- Average = 2.0/4 = 0.5
   TestForward2 (Loss, 0.5);

end MAE_Test;