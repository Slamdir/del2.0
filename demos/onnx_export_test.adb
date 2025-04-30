with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Del; use Del;
with Del.Model;
with Del.Data; use Del.Data;
with Del.ONNX;
with Del.Operators; use Del.Operators;
with Del.Loss; use Del.Loss;
with Del.Optimizers; use Del.Optimizers;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
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
      Linear_Layer : Linear_Access_T := new Linear_T;
      Relu_Layer : ReLU_Access_T := new ReLU_T;
   begin
      -- Initialize linear layer
      Linear_Layer.Initialize(100, 50);  -- Match input/output dimensions
      
      -- Add layers to model
      Model.Add_Layer(Func_Access_T(Linear_Layer));
      Model.Add_Layer(Func_Access_T(Relu_Layer));
      
      -- Add loss function
      Model.Add_Loss(Loss_Access_T'(new Mean_Square_Error_T));
      
      -- Add optimizer
      Model.Set_Optimizer(Optim_Access_T'(new SGD_T'(
         Create_SGD_T(Learning_Rate => 0.01, Weight_Decay => 0.0, Momentum => 0.9))));
   end Build_Simple_Network;
   
begin
   Put_Line("Starting ONNX export/import test...");
   
   -- Step 1: Create and train original model
   Put_Line("Building original model...");
   Build_Simple_Network(Original_Model);
   
   -- Create a dataset from the test data and load it into the model
   Put_Line("Setting up dataset for original model...");
   Original_Model.Set_Dataset(Del.Data.Create(Test_Data, Test_Labels));
   
   -- Train the model using the simplified API
   Put_Line("Training original model...");
   Original_Model.Train_Model(
      Num_Epochs => 5,
      Batch_Size => 2);  -- Adjusted batch size to be < 10
   
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
   
   -- Set the same dataset for the loaded model
   Put_Line("Setting up dataset for loaded model...");
   Loaded_Model.Set_Dataset(Del.Data.Create(Test_Data, Test_Labels));
   
   -- Add loss function and optimizer to loaded model before training
   Loaded_Model.Add_Loss(Loss_Access_T'(new Mean_Square_Error_T));
   Loaded_Model.Set_Optimizer(Optim_Access_T'(new SGD_T'(
      Create_SGD_T(Learning_Rate => 0.01, Weight_Decay => 0.0, Momentum => 0.9))));
   
   -- Step 5: Train the loaded model to ensure it works
   Put_Line("Training loaded model...");
   Loaded_Model.Train_Model(
      Num_Epochs => 5,
      Batch_Size => 2);
   
   Put_Line("Test completed successfully!");
exception
   when E : others =>
      Put_Line("Error: " & Exception_Message(E));
end ONNX_Export_Test;