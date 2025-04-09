with Del;
with Del.Operators;
with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Orka.Numerics.Doubles.Tensors.CPU; use Orka.Numerics.Doubles.Tensors.CPU;
with Orka.Numerics.Doubles.Tensors; use Orka.Numerics.Doubles.Tensors;
with Orka; use Orka;

procedure linear_test is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   
   -- Test dimensions
   Input_Size  : constant := 2;
   Output_Size : constant := 3;
   Batch_Size  : constant := 2;
   
   -- Linear layer test tensors
   Input_Data : D.Tensor_T := Ones((Batch_Size, Input_Size));  -- 2x2 input
   Gradient_Data : D.Tensor_T := Ones((Batch_Size, Output_Size));  -- 2x3 gradient
   Test_Result : D.Tensor_T := Zeros((Batch_Size, Output_Size)); -- Initialize with zeros
   
   -- Create layer and network
   L : DOp.Linear_T;
   Network : DMod.Model;
   Linear_Layer : DOp.Linear_Access_T;

   procedure Print_Shape(T : D.Tensor_T; Name : String) is
      S : D.Tensor_Shape_T := Shape(T);
   begin
      Put(Name & " shape: (");
      for I in S'Range loop
         if I > S'First then
            Put(", ");
         end if;
         Put(S(I)'Image);
      end loop;
      Put_Line(")");
   end Print_Shape;

begin
   Put_Line("=== Linear Layer Tests ===");
   
   -- Initialize the layer with weights and biases
   Put_Line("1. Initializing Linear Layer " & 
           Input_Size'Image & " -> " & Output_Size'Image & " nodes");
   L.Initialize(Input_Size, Output_Size);
   
   -- Print initial weights and biases
   declare
      Params : D.Params_T := L.Get_Params;
      Weights : D.Tensor_T := Params(0).all;
      Bias : D.Tensor_T := Params(1).all;
   begin
      Put_Line("Initial Weights:");
      Put_Line(Image(Weights));
      Print_Shape(Weights, "Weights");
      Put_Line("Initial Bias:");
      Put_Line(Image(Bias));
      Print_Shape(Bias, "Bias");
   end;
   
   Put_Line("");
   -- Test forward pass
   Put_Line("2. Testing Forward Pass");
   Put_Line("Input Data:");
   Put_Line(Image(Input_Data));
   Print_Shape(Input_Data, "Input");

   Put_Line("Computing forward pass...");
   begin
      Test_Result := L.Forward(Input_Data);
      Put_Line("Forward Pass Output:");
      Put_Line(Image(Test_Result));
      Print_Shape(Test_Result, "Output");
   exception
      when E : others =>
         Put_Line("Error in forward pass: " & Ada.Exceptions.Exception_Information(E));
   end;
   
   Put_Line("");
   -- Test backward pass
   Put_Line("3. Testing Backward Pass");
   Put_Line("Gradient Data:");
   Put_Line(Image(Gradient_Data));
   Print_Shape(Gradient_Data, "Gradient");
   
   begin
      Test_Result := L.Backward(Gradient_Data);
      Put_Line("Backward Pass Output (dL/dX):");
      Put_Line(Image(Test_Result));
      Print_Shape(Test_Result, "Backward Output");
   exception
      when E : others =>
         Put_Line("Error in backward pass: " & Ada.Exceptions.Exception_Information(E));
   end;
   
   -- Check accumulated gradients through Get_Params
   declare
      Params : D.Params_T := L.Get_Params;
   begin
      Put_Line("Parameters after backward pass:");
      Put_Line("Weights:");
      Put_Line(Image(Params(0).all));
      Print_Shape(Params(0).all, "Final Weights");
      Put_Line("Bias:");
      Put_Line(Image(Params(1).all));
      Print_Shape(Params(1).all, "Final Bias");
   end;
   
-- Test in network context
   Put_Line("4. Testing in Network Context");
   begin
      -- Create and initialize a new linear layer
      Linear_Layer := new DOp.Linear_T;
      Linear_Layer.Initialize(Input_Size, Output_Size);
      
      -- Debug print
      Put_Line("Created and initialized new layer");
      
      -- Add to network and initialize
      DMod.Add_Layer(Network, D.Func_Access_T(Linear_Layer));
      
      -- Debug print
      Put_Line("Added layer to network");
      
      -- Use same input data as individual test
      Put_Line("About to run network layers with input:");
      Put_Line(Image(Input_Data));
      Print_Shape(Input_Data, "Network Input");
      
      declare
         Network_Result : D.Tensor_T := Input_Data;
         Layer_Result : D.Tensor_T := Zeros((Batch_Size, Output_Size));  -- Initialize with proper dimensions
      begin
         Layer_Result := Network.Run_Layers(Network_Result);
         Put_Line("Network Forward Pass Result:");
         Put_Line(Image(Layer_Result));
         Print_Shape(Layer_Result, "Network Output");
      end;
   exception
      when E : others =>
         Put_Line("Error in network execution: " & Ada.Exceptions.Exception_Information(E));
         Put_Line("Last known good state: ");
         Put_Line("Input shape: " & Shape(Input_Data)(1)'Image & "," & Shape(Input_Data)(2)'Image);
   end;
end linear_test;