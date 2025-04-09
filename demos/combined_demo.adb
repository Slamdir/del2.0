with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;
with Ada.Containers.Vectors;
with Del; use Del;
with Del.Model; use Del.Model;
with Del.Operators; use Del.Operators;
with Del.Optimizers; use Del.Optimizers;
with Del.JSON; use Del.JSON;
with Del.ONNX; use Del.ONNX;
with Orka.Numerics.Doubles.Tensors; use Orka.Numerics.Doubles.Tensors;
with Orka.Numerics.Doubles.Tensors.CPU; use Orka.Numerics.Doubles.Tensors.CPU;
with Orka; use Orka;

procedure Combined_Demo is
-- Model and data declarations
   My_Model      : Del.Model.Model;
   Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- Per sample: 1 sample, 2 features
   Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 4);  -- Per sample: 1 sample, 4 classes
   Json_Filename : constant String := "demos/demo-data/spiral_3.json";
   Model_Path    : constant String := "bin/model.onnx";
   Batch_Size    : constant Positive := 10;  -- Process 10 samples per batch
   Num_Epochs    : constant Positive := 50;

   -- Optimizer
   Optim         : Del.Optimizers.SGD_T := Del.Optimizers.Create_SGD_T(
      Learning_Rate => 0.1, Weight_Decay => 0.1, Momentum => 0.9);

   -- Utility procedure to print tensor shape and values
   procedure Print_Tensor(T : Tensor_T; Name : String) is
      S : Tensor_Shape_T := Shape(T);
   begin
      Put_Line(Name & " shape: (");
      for I in S'Range loop
         Put(S(I)'Image);
         if I < S'Last then
            Put(", ");
         end if;
      end loop;
      Put_Line(")");
      Put_Line(Name & " values:");
      Put_Line(Image(T));
   end Print_Tensor;

   begin
   -- Print header with separator
   Put_Line("=============================================================");
   Put_Line("                 DEL: Deep Learning in Ada                   ");
   Put_Line("=============================================================");
   New_Line;

     -- Step 1: Load ONNX model
   Put_Line("STEP 1: LOADING ONNX MODEL");
   Put_Line("------------------------");
   Put_Line("Loading ONNX model from: " & Model_Path);
   if not Exists(Model_Path) then
      Put_Line("ERROR: ONNX file not found: " & Model_Path);
      return;
   end if;
   Load_ONNX_Model(My_Model, Model_Path);
   Put_Line("ONNX model loaded successfully.");
   New_Line;

   -- Step 2: Verify and load JSON data
   Put_Line("STEP 2: DATASET CONFIGURATION");
   Put_Line("---------------------------");
   Put_Line("Loading JSON data from: " & Json_Filename);
   if not Exists(Json_Filename) then
      Put_Line("ERROR: JSON file not found: " & Json_Filename);
      return;
   end if;
   Put_Line("JSON file located: " & Json_Filename);
   Put_Line("Data Shape (per sample): (" & Data_Shape(1)'Image & ", " & Data_Shape(2)'Image & ")");
   Put_Line("Target Shape (per sample): (" & Target_Shape(1)'Image & ", " & Target_Shape(2)'Image & ")");
   Put_Line("Batch Size:" & Batch_Size'Image);
   New_Line;

   -- Step 3: Train model with JSON data
   Put_Line("STEP 3: MODEL TRAINING");
   Put_Line("--------------------");
   Put_Line("Initiating training with JSON data...");
   
   -- Display training configuration
   Put_Line("Training configuration:");
   Put_Line("  Epochs:" & Num_Epochs'Image);
   Put_Line("  Batch size:" & Batch_Size'Image);
   
   -- Loading data
   Put_Line("Loading data from JSON file: " & Json_Filename);
   Put_Line("Starting training process...");
   
   -- Actual training call
   Put_Line("Executing actual training...");
   Train_Model_JSON
     (Self          => My_Model,
      JSON_File     => Json_Filename,
      Data_Shape    => Data_Shape,
      Target_Shape  => Target_Shape,
      Batch_Size    => Batch_Size,
      Num_Epochs    => Num_Epochs);
   
   Put_Line("Training completed successfully.");
   New_Line;

    -- Step 4: Test forward pass with sample data
   Put_Line("STEP 4: MODEL EVALUATION");
   Put_Line("----------------------");
   Put_Line("Testing forward pass with sample data...");
   declare
      Sample_Input   : Tensor_T := Ones((1, 2));  -- Matches JSON data and new ONNX input
      Forward_Result : Tensor_T := Zeros((1, 4)); -- Expected output shape
      Layers         : constant Layer_Vectors.Vector := My_Model.Get_Layers_Vector;
      Output1        : Tensor_T := Layers.Element(1).all.Forward(Sample_Input);  -- linear_1: (1, 2) -> (1, 50)
      Output2        : Tensor_T := Layers.Element(2).all.Forward(Output1);       -- relu_1: (1, 50) -> (1, 50)
      Output3        : Tensor_T := Layers.Element(3).all.Forward(Output2);       -- linear_2: (1, 50) -> (1, 4)
   begin
      Put_Line("Sample input:");
      Print_Tensor(Sample_Input, "Input");

      Put_Line("Total layers: " & Integer'Image(Integer(Layers.Length)));
      
      Put_Line("Layer 1 completed (Linear)");
      Print_Tensor(Output1, "Output from layer 1");
      Put_Line("Layer 2 completed (ReLU)");
      Print_Tensor(Output2, "Output from layer 2");
      Put_Line("Layer 3 completed (Linear)");
      Print_Tensor(Output3, "Output from layer 3");

      Forward_Result := Output3;
      Put_Line("Forward pass result:");
      Print_Tensor(Forward_Result, "Output");
   exception
      when E : others =>
         Put_Line("Error in forward pass: " & Exception_Information(E));
         raise;
   end;
   New_Line;

   
      -- Step 5: Apply optimizer steps
   Put_Line("STEP 5: OPTIMIZER APPLICATION");
   Put_Line("---------------------------");
   Put_Line("Applying optimizer steps...");
   for I in 1 .. 2 loop
      Put_Line("Optimizer Step " & I'Image);
      Optim.Step(My_Model.Get_Layers_Vector);
   end loop;
   Put_Line("Optimization steps completed.");
   New_Line;

    -- Step 6: Final forward pass after optimization
   Put_Line("STEP 6: MODEL EVALUATION (POST-OPTIMIZATION)");
   Put_Line("-----------------------------------------");
   Put_Line("Final forward pass after optimization...");
   declare
      Final_Input    : Tensor_T := Ones((1, 2));  -- Matches JSON data and new ONNX input
      Forward_Result : Tensor_T := Zeros((1, 4)); -- Expected output shape
      Layers         : constant Layer_Vectors.Vector := My_Model.Get_Layers_Vector;
      Output1        : Tensor_T := Layers.Element(1).all.Forward(Final_Input);  -- linear_1: (1, 2) -> (1, 50)
      Output2        : Tensor_T := Layers.Element(2).all.Forward(Output1);      -- relu_1: (1, 50) -> (1, 50)
      Output3        : Tensor_T := Layers.Element(3).all.Forward(Output2);      -- linear_2: (1, 50) -> (1, 4)
   begin
      Put_Line("Total layers: " & Integer'Image(Integer(Layers.Length)));
      
      Put_Line("Layer 1 completed (Linear)");
      Print_Tensor(Output1, "Output from layer 1");
      Put_Line("Layer 2 completed (ReLU)");
      Print_Tensor(Output2, "Output from layer 2");
      Put_Line("Layer 3 completed (Linear)");
      Print_Tensor(Output3, "Output from layer 3");

      Forward_Result := Output3;
      Put_Line("Final forward pass result:");
      Print_Tensor(Forward_Result, "Final Output");
   exception
      when E : others =>
         Put_Line("Error in final forward pass: " & Exception_Information(E));
         raise;
   end;
   New_Line;

   -- Success message with separator
   Put_Line("=============================================================");
   Put_Line("             Demo completed successfully!                    ");
   Put_Line("=============================================================");

exception
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));

end Combined_Demo;