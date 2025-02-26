with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Del; use Del;
with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Operators; use Del.Operators;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Ada.Directories; use Ada.Directories;

procedure Json_Test is
begin
   Put_Line("=== Starting JSON Training Test ===");
   
   declare
      My_Model      : Del.Model.Model;
      Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- 1 sample, 2 features
      Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 4);  -- 1 sample, 4 classes
      Json_Filename : constant String := "bin/initial_testing.json";
      Batch_Size    : constant Positive := 10;
      Num_Epochs    : constant Positive := 1;
   begin
      -- Log initial setup
      Put_Line("Model Configuration:");
      Put_Line("  Data Shape: (1, 2)");
      Put_Line("  Target Shape: (1, 4)");
      Put_Line("  JSON File: " & Json_Filename);
      Put_Line("  Batch Size:" & Batch_Size'Image);
      Put_Line("  Number of Epochs:" & Num_Epochs'Image);

      -- Verify JSON file
      if not Exists(Json_Filename) then
         Put_Line("ERROR: JSON file not found: " & Json_Filename);
         return;
      else
         Put_Line("JSON file located: " & Json_Filename);
      end if;

      -- Build the model
      Put_Line("Constructing model...");
      declare
         Layer1 : Func_Access_T := new Linear_T;
         Layer2 : Func_Access_T := new ReLU_T;
      begin
         Put_Line("  Initializing Linear layer (Input: 2, Output: 5)...");
         Linear_T(Layer1.all).Initialize(2, 5);  -- 2 inputs -> 5 outputs
         Del.Model.Add_Layer(My_Model, Layer1);
         Put_Line("  Linear layer added.");

         Put_Line("  Adding ReLU activation layer...");
         Del.Model.Add_Layer(My_Model, Layer2);
         Put_Line("  ReLU layer added.");

         Put_Line("Model Architecture: Linear (2 -> 5) -> ReLU");
      end;

      -- Train the model
      Put_Line("Initiating training...");
      Del.Model.Train_Model_JSON
        (Self          => My_Model,
         JSON_File     => Json_Filename,
         Data_Shape    => Data_Shape,
         Target_Shape  => Target_Shape,
         Batch_Size    => Batch_Size,
         Num_Epochs    => Num_Epochs);

      Put_Line("Training completed successfully!");
   end;
   
exception
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));
end Json_Test;