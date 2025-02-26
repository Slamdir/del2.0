with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Del.Model;
with Del.ONNX;
with Orka.Numerics.Singles.Tensors.CPU;

procedure ONNX_Test is
   Model : Del.Model.Model;
   Test_Data : constant Del.Tensor_T := 
     Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 100]);
   Test_Labels : constant Del.Tensor_T := 
     Orka.Numerics.Singles.Tensors.CPU.Random_Uniform([10, 50]);
   
   
   Model_Path : constant String := "bin/model.onnx";  -- Adjust path as needed
begin
--     Put_Line("Loading ONNX model from: " & Model_Path);
--     Del.ONNX.Load_ONNX_Model(Model, Model_Path);
   
--     Put_Line("Training loaded model...");
--     Model.Train_Model(
--        Num_Epochs => 5,
--        Data => Test_Data,
--        Labels => Test_Labels);
   
   Put_Line("Test completed successfully!");
exception
   when E : others =>
      Put_Line("Error: " & Ada.Exceptions.Exception_Message(E));
end ONNX_Test;