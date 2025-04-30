with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Del; use Del;
with Del.Model;
with Del.Data; use Del.Data;
with Del.JSON; use Del.JSON;
with Del.Optimizers; use Del.Optimizers;
with Del.Loss; use Del.Loss;
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
      Json_Filename : constant String := "demos/demo-data/spiral_3.json";
      Batch_Size    : constant Positive := 10;
      Num_Epochs    : constant Positive := 10;  -- Increased epochs for better accuracy
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
         Layer1 : Linear_Access_T := new Linear_T;      -- First linear layer
         Layer2 : ReLU_Access_T := new ReLU_T;          -- First activation
         Layer3 : Linear_Access_T := new Linear_T;      -- Second linear layer 
         Layer4 : SoftMax_Access_T := new SoftMax_T;    -- Output activation
      begin
         Put_Line("  Initializing first Linear layer (Input: 2, Output: 5)...");
         Layer1.Initialize(2, 5);  -- 2 inputs -> 5 hidden units
         My_Model.Add_Layer(Func_Access_T(Layer1));
         Put_Line("  First Linear layer added.");

         Put_Line("  Adding ReLU activation layer...");
         My_Model.Add_Layer(Func_Access_T(Layer2));
         Put_Line("  ReLU layer added.");

         Put_Line("  Initializing second Linear layer (Input: 5, Output: 4)...");
         Layer3.Initialize(5, 4);  -- 5 hidden -> 4 outputs (matching target shape)
         My_Model.Add_Layer(Func_Access_T(Layer3));
         Put_Line("  Second Linear layer added.");

         Put_Line("  Adding Softmax activation layer...");
         My_Model.Add_Layer(Func_Access_T(Layer4));
         Put_Line("  Softmax layer added.");

         -- Add loss function
         Put_Line("  Adding Cross Entropy loss function...");
         My_Model.Add_Loss(Loss_Access_T'(new Cross_Entropy_T));
         Put_Line("  Loss function added.");
         
         -- Add optimizer using the provided Create_SGD_T function
         Put_Line("  Setting SGD optimizer (LR=0.01, Momentum=0.9)...");
         My_Model.Set_Optimizer(Optim_Access_T'(new SGD_T'(
           Create_SGD_T(Learning_Rate => 0.01, Weight_Decay => 0.0, Momentum => 0.9))));
         Put_Line("  Optimizer set.");

         Put_Line("Model Architecture: Linear (2 -> 5) -> ReLU -> Linear (5 -> 4) -> Softmax");
      end;

      -- Load data from JSON (now using the model's wrapper function)
      Put_Line("Loading data from JSON...");
      My_Model.Load_Data_From_JSON
        (JSON_File     => Json_Filename,
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
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));
end Json_Test;