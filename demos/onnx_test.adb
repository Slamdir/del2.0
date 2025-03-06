with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Del.Model;
with Del.ONNX;
with Del.Operators; use Del.Operators;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure ONNX_test is
   Model : Del.Model.Model;
   Model_Path : constant String := "bin/model.onnx";
begin
   Put_Line("ONNX Structure Test");
   Put_Line("==================");
   
   -- Load the ONNX model
   Put_Line("Loading ONNX model from: " & Model_Path);
   Del.ONNX.Load_ONNX_Model(Model, Model_Path);
   Put_Line("Model loaded successfully");
   
   -- Display model structure
   declare
      Layers : constant Del.Layer_Vectors.Vector := Model.Get_Layers_Vector;
      Layer_Count : constant Natural := Natural(Layers.Length);
   begin
      Put_Line("Model has" & Layer_Count'Image & " layers");
      
      for I in 1 .. Layer_Count loop
         if Layers(I).all in Linear_T'Class then
            Put_Line("Layer" & I'Image & ": Linear");
         elsif Layers(I).all in ReLU_T'Class then
            Put_Line("Layer" & I'Image & ": ReLU");
         elsif Layers(I).all in SoftMax_T'Class then
            Put_Line("Layer" & I'Image & ": SoftMax");
         else
            Put_Line("Layer" & I'Image & ": Unknown");
         end if;
      end loop;
   end;
   
   -- Export the model without modification
   Put_Line("Exporting model to ONNX...");
   Del.ONNX.Save_ONNX_Model(Model, "exported_model.onnx");
   Put_Line("Model exported successfully to exported_model.onnx");
   
   Put_Line("Test completed successfully!");
exception
   when E : others =>
      Put_Line("Error: " & Exception_Message(E));
      Put_Line("Exception info: " & Exception_Information(E));
end ONNX_test;