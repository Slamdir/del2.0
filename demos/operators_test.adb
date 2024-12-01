with Ada.Text_IO; use Ada.Text_IO;
with Ada.Float_Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka.Numerics.Singles; use Orka.Numerics.Singles;
with Del;
with Del.Operators;
with Del.Initializers;

procedure Operators_Test is

   package D    renames Del;
   package DOp  renames Del.Operators;
   package DI   renames Del.Initializers;

   -- Input tensor X
   X : D.Tensor_T := To_Tensor ([1.0, -2.0, 3.0,
                                 4.0, -5.0, 6.0,
                                 7.0, -8.0, 9.0], [3, 3]);

   -- Gradient tensor Dy (used for backpropagation)
   Dy : D.Tensor_T := To_Tensor ([0.1, 0.2, 0.3,
                                  0.4, 0.5, 0.6,
                                  0.7, 0.8, 0.9], [3, 3]);

   -- Instantiate operator objects
   L : DOp.Linear_T;
   R : DOp.ReLU_T;
   M : DOp.SoftMax_T;

   -- Output tensors initialized with appropriate dimensions
   Linear_Output  : D.Tensor_T := Zeros ([3, 3]);
   ReLU_Output    : D.Tensor_T := Zeros ([3, 3]);
   SoftMax_Output : D.Tensor_T := Zeros ([3, 3]);

   -- Backward output tensors
   Linear_Backward_Output : D.Tensor_T := Zeros ([3, 3]);
   ReLU_Backward_Output   : D.Tensor_T := Zeros ([3, 3]);
   SoftMax_Backward_Output : D.Tensor_T := Zeros ([3, 3]);

   -- Procedure to print Tensor_T
   procedure Print_Tensor (T : D.Tensor_T) is
      use Ada.Float_Text_IO;

      Rows  : Positive := T.Shape (1);
      Cols  : Positive := T.Shape (2);
   begin
      Put_Line ("[");
      for I in 1 .. Rows loop
         Put (" [");
         declare
            Row_Slice : D.Tensor_T := T (I);  -- Assuming slicing is possible with `T(I)`
         begin
            for J in 1 .. Cols loop
               declare
                  -- Storing in an intermediate variable to resolve ambiguity
                  Element_F32 : Float_32 := Row_Slice.Get(J);  -- Get element of type Float_32
                  Element     : Standard.Float := Standard.Float (Element_F32);  -- Convert explicitly to Standard.Float
               begin
                  Put (Item => Element, Aft => 6, Exp => 0);
                  if J < Cols then
                     Put (", ");
                  end if;
               end;
            end loop;
         end;
         Put ("]");
         if I < Rows then
            Put_Line (",");
         else
            Put_Line ("");
         end if;
      end loop;
      Put_Line ("]");
   end Print_Tensor;

begin

   Put_Line ("Testing Del.Operators Package");
   Put_Line ("--------------------------------");

   -- Test Linear_T Forward and Backward
   Put_Line ("Testing Linear_T Forward:");
   Linear_Output := DOp.Forward (L, X);
   Put_Line ("Linear_T Forward Output:");
   Print_Tensor (Linear_Output);
   Put_Line ("");

   Put_Line ("Testing Linear_T Backward:");
   Linear_Backward_Output := DOp.Backward (L, Dy);
   Put_Line ("Linear_T Backward Output:");
   Print_Tensor (Linear_Backward_Output);
   Put_Line ("");

   -- Test ReLU_T Forward and Backward
   Put_Line ("Testing ReLU_T Forward:");
   ReLU_Output := DOp.Forward (R, X);
   Put_Line ("ReLU_T Forward Output:");
   Print_Tensor (ReLU_Output);
   Put_Line ("");

   Put_Line ("Testing ReLU_T Backward:");
   ReLU_Backward_Output := DOp.Backward (R, Dy);
   Put_Line ("ReLU_T Backward Output:");
   Print_Tensor (ReLU_Backward_Output);
   Put_Line ("");

   -- Test SoftMax_T Forward and Backward
   Put_Line ("Testing SoftMax_T Forward:");
   SoftMax_Output := DOp.Forward (M, X);
   Put_Line ("SoftMax_T Forward Output:");
   Print_Tensor (SoftMax_Output);
   Put_Line ("");

   Put_Line ("Testing SoftMax_T Backward:");
   SoftMax_Backward_Output := DOp.Backward (M, Dy);
   Put_Line ("SoftMax_T Backward Output:");
   Print_Tensor (SoftMax_Backward_Output);
   Put_Line ("");

end Operators_Test;
