with Ada.Text_IO;
with DL_Linear;
use Ada.Text_IO;
use DL_Linear;

procedure Test_DL_Linear is

   Input    : constant Float_Array (1 .. 3) := (1.0, 2.0, 3.0);
   Layer    : Linear_Layer (Input_Size => 3, Output_Size => 2);
   Output   : Float_Array (1 .. 2);
   d_Output : constant Float_Array (1 .. 2) := (0.1, -0.1);


   procedure Print_Array (Arr : Float_Array) is
   begin
      for Value of Arr loop
         Put_Line (Float'Image (Value));
      end loop;
   end Print_Array;

begin
   Put_Line ("Testing DL_Linear Package...");


   Initialize_Layer (Layer);
   Put_Line ("Layer Initialized with Random Weights and Zero Bias");


   Put_Line ("Initial Weights:");
   for I in Layer.Weights'Range (1) loop
      for J in Layer.Weights'Range (2) loop
         Put (Float'Image (Layer.Weights (I, J)) & " ");
      end loop;
      New_Line;
   end loop;

   Put_Line ("Initial Biases:");
   Print_Array (Layer.Bias);


   Output := Forward (Input, Layer);
   Put_Line ("Forward Pass Output:");
   Print_Array (Output);


   Put_Line ("Performing Backward Pass...");
   Backward (Input, d_Output, Layer);


   Put_Line ("Updated Weights:");
   for I in Layer.Weights'Range (1) loop
      for J in Layer.Weights'Range (2) loop
         Put (Float'Image (Layer.Weights (I, J)) & " ");
      end loop;
      New_Line;
   end loop;

   Put_Line ("Updated Biases:");
   Print_Array (Layer.Bias);

   Put_Line ("Testing Complete.");
end Test_DL_Linear;
