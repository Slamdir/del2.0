with Ada.Text_IO;               use Ada.Text_IO;
with Ada.Exceptions;            use Ada.Exceptions;
with Ada.Directories;           use Ada.Directories;

with Del.Model;
with Del.ONNX;

with Orka.Numerics.Singles.Tensors.CPU;

procedure ONNX_Tests is

   ----------------------------------------------------------------------------
   -- Sub-Test #1: Check if the ONNX model is found and can be loaded
   ----------------------------------------------------------------------------
   procedure Test_Load_Model is
      Model_Path : constant String := "bin/model.onnx";  
      My_Model   : Del.Model.Model;
   begin
      Put_Line("--- Test_Load_Model ---");

      if not Exists(Model_Path) then
         Put_Line("FAIL: model.onnx not found at: " & Model_Path);
         return;
      else
         Put_Line("File found: " & Model_Path);
      end if;

      begin
         Del.ONNX.Load_ONNX_Model(My_Model, Model_Path);
         Put_Line("PASS: Model loaded successfully into My_Model.");
      exception
         when E : others =>
            Put_Line("FAIL: Exception loading model: " & Exception_Message(E));
      end;
   end Test_Load_Model;

   ----------------------------------------------------------------------------
   -- Sub-Test #2: Verify the model’s training routine for N epochs
   ----------------------------------------------------------------------------
   procedure Test_Train_Model is
      Model_Path : constant String := "bin/model.onnx";
      My_Model   : Del.Model.Model;
      Num_Epochs : constant Positive := 3;

      -- We create random data with shape 10×100, matching 100 inputs
      -- and random labels with shape 10×50, matching 50 outputs.
      Test_Data   : Del.Tensor_T :=
        Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 100]);
      Test_Labels : Del.Tensor_T :=
        Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 50]);
   begin
      Put_Line("--- Test_Train_Model ---");

      if not Exists(Model_Path) then
         Put_Line("FAIL: Cannot train because " & Model_Path & " does not exist.");
         return;
      end if;

      begin
         Del.ONNX.Load_ONNX_Model(My_Model, Model_Path);
         Put_Line("Model loaded for training: " & Model_Path);
      exception
         when E : others =>
            Put_Line("FAIL: Exception loading model for training: " & Exception_Message(E));
            return;
      end;

      -- Attempt training
      begin
         My_Model.Train_Model(
           Num_Epochs => Num_Epochs,
           Data       => Test_Data,
           Labels     => Test_Labels
         );
         Put_Line("PASS: Successfully trained ONNX model for " & Num_Epochs'Image & " epochs.");
      exception
         when E : others =>
            Put_Line("FAIL: Exception during training: " & Exception_Message(E));
      end;
   end Test_Train_Model;

   ----------------------------------------------------------------------------
   -- Sub-Test #3: Check proper exception handling for missing or invalid file
   ----------------------------------------------------------------------------
   procedure Test_Exception_Handling is
      Invalid_Path : constant String := "bin/does_not_exist.onnx";
      My_Model     : Del.Model.Model;
   begin
      Put_Line("--- Test_Exception_Handling ---");

      if Exists(Invalid_Path) then
         Put_Line("WARNING: " & Invalid_Path & " actually exists; rename or remove it.");
      end if;

      begin
         Del.ONNX.Load_ONNX_Model(My_Model, Invalid_Path);
         Put_Line("FAIL: Expected an exception for missing/invalid file, but got none.");
      exception
         when E : others =>
            Put_Line("PASS: Caught expected exception for missing file: " & Exception_Message(E));
      end;
   end Test_Exception_Handling;

   ----------------------------------------------------------------------------
   -- Sub-Test #4: Integrate the ONNX model in a mini “pipeline”
   ----------------------------------------------------------------------------
   procedure Test_ONNX_Pipeline is
      Model_Path : constant String := "bin/model.onnx";
      My_Model   : Del.Model.Model;

      
      Pipeline_Data   : Del.Tensor_T :=
        Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([5, 100]);
      Pipeline_Labels : Del.Tensor_T :=
        Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([5, 50]);
   begin
      Put_Line("--- Test_ONNX_Pipeline ---");

      if not Exists(Model_Path) then
         Put_Line("FAIL: " & Model_Path & " not found for pipeline test.");
         return;
      end if;

      begin
         Del.ONNX.Load_ONNX_Model(My_Model, Model_Path);
         Put_Line("Model loaded for pipeline: " & Model_Path);
      exception
         when E : others =>
            Put_Line("FAIL: Could not load model in pipeline: " & Exception_Message(E));
            return;
      end;

      
      begin
         My_Model.Train_Model(
           Num_Epochs => 5,
           Data       => Pipeline_Data,
           Labels     => Pipeline_Labels
         );
         Put_Line("PASS: ONNX model integrated in pipeline, trained for 2 epochs.");
      exception
         when E : others =>
            Put_Line("FAIL: Error in pipeline integration: " & Exception_Message(E));
      end;
   end Test_ONNX_Pipeline;

begin
   Put_Line("=== ONNX TESTS BEGIN ===");

   Test_Load_Model;
   Test_Train_Model;
   Test_Exception_Handling;
   Test_ONNX_Pipeline;

   Put_Line("=== ONNX TESTS END ===");
end ONNX_Tests;
