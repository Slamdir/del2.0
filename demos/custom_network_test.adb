with Del;
with Del.Operators;
with Del.Model;
with Del.Loss;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka;

procedure Custom_Network_Test is

   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DLoss renames Del.Loss;

   -- Input and expected output tensors
   X        : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
   Expected : D.Tensor_T := To_Tensor([0.0, 1.0, 0.0, 1.0], [2, 2]);  -- One-hot expected output

   -- Define a custom network
   Network : DMod.Model;

   -- Attach a loss function
   Loss : DLoss.Cross_Entropy_T;

   -- Output tensors with explicit initialization
   Forward_Output : D.Tensor_T := Zeros((2, 2));  -- Match dimensions with output of softmax layer
   Gradients      : D.Tensor_T := Zeros((2, 2));

   -- Procedure to print a tensor
   procedure Print_Tensor(T : D.Tensor_T) is
      Length : constant Integer := Shape(T)(1) * Shape(T)(2);
   begin
      for I in 1 .. Length loop
         declare
            Element_F32 : constant Orka.Float_32 := T.Get(I);
            Element     : constant Float := Float(Element_F32);
         begin
            Put_Line("Element[" & I'Image & "] = " & Float'Image(Element));
         end;
      end loop;
   end Print_Tensor;

   -- Procedure to print tensor shape
   procedure Print_Tensor_Shape(T : D.Tensor_T) is
   begin
      Put_Line("Tensor Shape: " & Shape(T)'Image);
   end Print_Tensor_Shape;

begin
   -- Build the custom network
   Put_Line("Building Custom Network...");
   DMod.Add_Layer(Network, new DOp.Linear_T);
   DMod.Add_Layer(Network, new DOp.ReLU_T);
   DMod.Add_Layer(Network, new DOp.SoftMax_T);

   -- Run forward propagation using the provided procedure
   Put_Line("Running Forward Propagation...");

   -- Print input tensor shape
   Put_Line("Input Tensor Shape:");
   Print_Tensor_Shape(X);

   -- Run the network using the provided Run_Layers procedure
   Forward_Output := DMod.Run_Layers(Network, X);

   -- Print the forward output
   Put_Line("Forward Output:");
   Print_Tensor(Forward_Output);

   -- **Additional Debugging**: Print output tensor shape before loss calculation
   Put_Line("Forward Output Tensor Shape:");
   Print_Tensor_Shape(Forward_Output);

   -- Compute gradients using the loss function
   Put_Line("Computing Gradients...");

   -- **Additional Debugging**: Print expected and forward output shapes before computing loss
   Put_Line("Expected Tensor Shape:");
   Print_Tensor_Shape(Expected);
   Put_Line("Forward Output Tensor Shape for Loss:");
   Print_Tensor_Shape(Forward_Output);

   -- Ensure the shapes match
   Gradients := Loss.Backward(Expected, Forward_Output);

   -- Print gradients
   Put_Line("Gradients:");
   Print_Tensor(Gradients);

exception
   when Constraint_Error =>
      Put_Line("Constraint Error occurred. Check tensor initialization and dimensions.");

end Custom_Network_Test;
