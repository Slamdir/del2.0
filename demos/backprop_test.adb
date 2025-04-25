with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;
with Ada.Containers.Vectors;
with Del; use Del;
with Del.Model; use Del.Model;
with Del.Operators; use Del.Operators;
with Del.Optimizers; use Del.Optimizers;
with Del.JSON; use Del.JSON;
with Del.Loss; use Del.Loss;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure backprop_test is
-- Model and data declarations
   My_Model      : Del.Model.Model;
   Linear_Layer  : Linear_Access_T;
   ReLU_Layer    : ReLU_Access_T;
   Softmax_Layer : Softmax_Access_T;
   Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- Per sample: 1 sample, 2 features
   Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 3);  -- Per sample: 1 sample, 3 classes
   Json_Filename : constant String := "demos/demo-data/spiral_3_6.json";
   Output_File : constant String := "demos\output\model_output.json";
   Batch_Size    : constant Positive := 20;  -- Process 10 samples per batch
   Num_Epochs    : constant Positive := 200;

   Hidden_Units  : constant Positive := 100;
   Num_Classes   : constant Positive := 3;

   Optimizer     : Optim_Access_T := new SGD_T'(Create_SGD_T(
      Learning_Rate => 0.6, Weight_Decay => 0.00001, Momentum => 0.9));

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

   -- Step 1: Create Model and add layers
   Linear_Layer := new Linear_T;
   Linear_Layer.Initialize(2, Hidden_Units);
   My_Model.Add_Layer(Del.Func_Access_T(Linear_Layer));

   ReLU_Layer := new ReLU_T;
   My_Model.Add_Layer(Del.Func_Access_T(ReLU_Layer));

   Linear_Layer := new Linear_T;
   Linear_Layer.Initialize(Hidden_Units, Num_Classes);
   My_Model.Add_Layer(Del.Func_Access_T(Linear_Layer));

   Softmax_Layer := new Softmax_T;
   My_Model.Add_Layer(Del.Func_Access_T(Softmax_Layer));

   -- Add loss function
   My_Model.Set_Optimizer(Optimizer);
   My_Model.Add_Loss(new Cross_Entropy_T); 
   
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
   
   -- Load data from JSON
   Put_Line("Loading data from JSON file: " & Json_Filename);
   My_Model.Load_Data_From_JSON(
      JSON_File     => Json_Filename,
      Data_Shape    => Data_Shape,
      Target_Shape  => Target_Shape);
   

   -- Train the model using the loaded data
   Put_Line("Starting training process...");
   My_Model.Train_Model(
      Batch_Size    => Batch_Size,
      Num_Epochs    => Num_Epochs);
   
   Put_Line("Training completed successfully.");
   New_Line;

   Put_Line("Exporting model output to JSON...");
   My_Model.Export_To_JSON(Output_File);
   Put_Line("Export complete. Output written to: " & Output_File);

   -- Success message with separator
   Put_Line("=============================================================");
   Put_Line("             Demo completed successfully!                    ");
   Put_Line("=============================================================");

exception
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));


end backprop_test;