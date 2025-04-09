with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Del; use Del;
with Del.Model;
with Del.Data; use Del.Data;
with Del.YAML; use Del.YAML;  -- Use Del.YAML instead of Del.JSON
with Del.Operators; use Del.Operators;
with Orka.Numerics.Doubles.Tensors; use Orka.Numerics.Doubles.Tensors;
with Ada.Directories; use Ada.Directories;

procedure YAML_Test is
begin
   Put_Line("=== Starting YAML Training Test ===");
   
   declare
      My_Model      : Del.Model.Model;
      Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- 1 sample, 2 features
      Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 4);  -- 1 sample, 4 classes
      YAML_Filename : constant String := "bin/initial_testing.yaml";
      Batch_Size    : constant Positive := 10;
      Num_Epochs    : constant Positive := 1;
   begin
      -- Log initial setup
      Put_Line("Model Configuration:");
      Put_Line("  Data Shape: (1, 2)");
      Put_Line("  Target Shape: (1, 4)");
      Put_Line("  YAML File: " & YAML_Filename);
      Put_Line("  Batch Size:" & Batch_Size'Image);
      Put_Line("  Number of Epochs:" & Num_Epochs'Image);

      -- Verify YAML file
      if not Exists(YAML_Filename) then
         Put_Line("ERROR: YAML file not found: " & YAML_Filename);
         return;
      else
         Put_Line("YAML file located: " & YAML_Filename);
      end if;

      -- Build the model
      Put_Line("Constructing model...");
      declare
         Layer1 : Func_Access_T := new Linear_T;
         Layer2 : Func_Access_T := new ReLU_T;
      begin
         Put_Line("  Initializing Linear layer (Input: 2, Output: 5)...");
         Linear_T(Layer1.all).Initialize(2, 5);  -- 2 inputs -> 5 outputs
         My_Model.Add_Layer(Layer1);
         Put_Line("  Linear layer added.");

         Put_Line("  Adding ReLU activation layer...");
         My_Model.Add_Layer(Layer2);
         Put_Line("  ReLU layer added.");

         Put_Line("Model Architecture: Linear (2 -> 5) -> ReLU");
      end;

      -- Load data from YAML (using the model's wrapper function)
      Put_Line("Loading data from YAML...");
      My_Model.Load_Data_From_YAML
        (YAML_File     => YAML_Filename,
         Data_Shape    => Data_Shape,
         Target_Shape  => Target_Shape);

      -- Train the model using the simplified training function
      Put_Line("Initiating training...");
      My_Model.Train_Model
        (Batch_Size => Batch_Size,
         Num_Epochs => Num_Epochs);

      Put_Line("Training completed successfully!");
   end;
   
exception
   when E : YAML_Parse_Error =>
      Put_Line("ERROR: YAML parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));
end YAML_Test;