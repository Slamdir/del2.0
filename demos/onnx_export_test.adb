with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Del.Model;
with Del.ONNX;
with Del.Operators; use Del.Operators;
with Orka.Numerics.Singles.Tensors.CPU;

procedure ONNX_Export_Test is
   -- Create two models - one for saving, one for loading
   Original_Model : Del.Model.Model;
   Loaded_Model : Del.Model.Model;
   
   -- Test data
   Test_Data : constant Del.Tensor_T := 
     Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 100]);
   Test_Labels : constant Del.Tensor_T := 
     Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 50]);
   
   -- Model paths
   Export_Path : constant String := "bin/exported_model.onnx";
   
   -- Helper procedure to create a simple network
   procedure Build_Simple_Network(Model : in out Del.Model.Model) is
      Linear_Layer : Linear_T;
      Relu_Layer : ReLU_T;
   begin
      -- Initialize linear layer
      Linear_Layer.Initialize(100, 50);  -- Match input/output dimensions
      
      -- Add layers to model
      Model.Add_Layer(new Linear_T'(Linear_Layer));
      Model.Add_Layer(new ReLU_T'(Relu_Layer));
   end Build_Simple_Network;
   
begin
   Put_Line("Starting ONNX export/import test...");
   
   -- Step 1: Create and train original model
   Put_Line("Building original model...");
   Build_Simple_Network(Original_Model);
   
   Put_Line("Training original model...");
   Original_Model.Train_Model(
      Num_Epochs => 5,
      Data => Test_Data,
      Labels => Test_Labels);
   
   -- Step 2: Export the trained model
   Put_Line("Exporting model to: " & Export_Path);
   Original_Model.Export_ONNX(Export_Path);
   
   -- Step 3: Load the exported model
   Put_Line("Loading exported model from: " & Export_Path);
   Del.ONNX.Load_ONNX_Model(Loaded_Model, Export_Path);
   
   -- Step 4: Test the loaded model
   Put_Line("Testing loaded model...");
   declare
      Original_Output : Del.Tensor_T := Original_Model.Run_Layers(Test_Data);
      Loaded_Output : Del.Tensor_T := Loaded_Model.Run_Layers(Test_Data);
   begin
      -- Compare outputs (you might want to add more detailed comparison)
      Put_Line("Original model output shape: " & 
               Original_Output.Shape(1)'Image & "," & 
               Original_Output.Shape(2)'Image);
      Put_Line("Loaded model output shape: " & 
               Loaded_Output.Shape(1)'Image & "," & 
               Loaded_Output.Shape(2)'Image);
   end;
   
   -- Step 5: Train the loaded model to ensure it works
   Put_Line("Training loaded model...");
   Loaded_Model.Train_Model(
      Num_Epochs => 5,
      Data => Test_Data,
      Labels => Test_Labels);
   
   Put_Line("Test completed successfully!");
exception
   when E : others =>
      Put_Line("Error: " & Ada.Exceptions.Exception_Message(E));
end ONNX_Export_Test;