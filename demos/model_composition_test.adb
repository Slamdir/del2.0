with Ada.Text_IO; use Ada.Text_IO;
with Ada.Float_Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Del;
with Del.Operators;
with Del.Loss;
with Del.Model;

procedure Model_Composition_Test is

   package D       renames Del;
   package DOp     renames Del.Operators;
   package DLoss   renames Del.Loss;
   package DMod    renames Del.Model;

   -- Input tensor for testing the model
   Input : D.Tensor_T := To_Tensor ([1.0, -2.0, 3.0,
                                     4.0, -5.0, 6.0,
                                     7.0, -8.0, 9.0], [3, 3]);

   -- Expected output tensor for loss calculation
   Expected : constant D.Tensor_T := To_Tensor ([1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0], [3, 3]);

   -- Instantiate operator and model objects
   Linear_Layer  : DOp.Linear_T;
   ReLU_Layer    : DOp.ReLU_T;
   SoftMax_Layer : DOp.SoftMax_T;
   Cross_Entropy : DLoss.Cross_Entropy_T;

   Model         : DMod.Model;

   -- Model output tensor and loss value
   Model_Output  : D.Tensor_T := Zeros (Input.Shape);
   Loss_Value    : D.Element_T;

   -- Procedure to print Tensor_T
   procedure Print_Tensor (T : D.Tensor_T) is
      use Ada.Float_Text_IO;

      Rows  : constant Positive := T.Shape (1);
      Cols  : constant Positive := T.Shape (2);
   begin
      Put_Line ("[");
      for I in 1 .. Rows loop
         Put (" [");
         declare
            Row_Slice : constant D.Tensor_T := T (I);
         begin
            for J in 1 .. Cols loop
               declare
                  Element_F32 : constant Float_32 := Row_Slice.Get (J);
                  Element     : constant Standard.Float := Standard.Float (Element_F32);
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

   Put_Line ("Testing Del.Model Package: Model Composition Test");
   Put_Line ("--------------------------------");

   -- Add Layers to Model
   Put_Line ("Adding Linear, ReLU, and SoftMax Layers to the Model:");
   DMod.Add_Layer (Model, new DOp.Linear_T'(Linear_Layer));
   DMod.Add_Layer (Model, new DOp.ReLU_T'(ReLU_Layer));
   DMod.Add_Layer (Model, new DOp.SoftMax_T'(SoftMax_Layer));
   Put_Line ("Layers added successfully.");
   Put_Line ("");

   -- Run the Model Layers with the Input Tensor
   Put_Line ("Running the Model with the Input Tensor:");
   Model_Output := DMod.Run_Layers (Model, Input);
   Put_Line ("Model Output:");
   Print_Tensor (Model_Output);
   Put_Line ("");

   -- Add Loss Function to the Model
   Put_Line ("Adding Cross-Entropy Loss Function to the Model:");
   DMod.Add_Loss (Model, new DLoss.Cross_Entropy_T'(Cross_Entropy));
   Put_Line ("Loss function added successfully.");
   Put_Line ("");

   -- Calculate Loss using the Model Output and Expected Tensor
   Put_Line ("Calculating Cross-Entropy Loss:");
   Loss_Value := Cross_Entropy.Forward (Expected, Model_Output);
   Put_Line ("Cross-Entropy Loss Value: " & Float'Image (Float (Loss_Value)));
   Put_Line ("");

end Model_Composition_Test;
