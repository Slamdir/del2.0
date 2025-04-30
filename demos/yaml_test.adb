with Ada.Text_IO; use Ada.Text_IO; 
with Ada.Exceptions; use Ada.Exceptions; 
with Del; use Del; 
with Del.Model; 
with Del.Data; use Del.Data; 
with Del.YAML; use Del.YAML;
with Del.Optimizers; use Del.Optimizers;
with Del.Loss; use Del.Loss;
with Del.Operators; use Del.Operators;  -- Added for access to layer types
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors; 
with Ada.Directories; use Ada.Directories; 
 
procedure YAML_Test is 
begin 
   Put_Line("=== Starting YAML Training Test ==="); 
    
   declare 
      My_Model      : Del.Model.Model; 
      Data_Shape    : constant Tensor_Shape_T := [1 => 1, 2 => 2];  -- Fixed syntax with []
      Target_Shape  : constant Tensor_Shape_T := [1 => 1, 2 => 3];  -- Fixed syntax with []
      YAML_Filename : constant String := "demos/demo-data/initial_testing.yaml";
      Batch_Size    : constant Positive := 10; 
      Num_Epochs    : constant Positive := 1; 
   begin 
      -- Log initial setup 
      Put_Line("Model Configuration:"); 
      Put_Line("  Data Shape: (1, 2)"); 
      Put_Line("  Target Shape: (1, 3)");
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
         Layer1 : Linear_Access_T := new Linear_T;  -- Use correct type Linear_Access_T
         Layer2 : ReLU_Access_T := new ReLU_T;      -- Use correct type ReLU_Access_T
         Layer3 : Linear_Access_T := new Linear_T;  -- Use correct type Linear_Access_T
         Layer4 : SoftMax_Access_T := new SoftMax_T;  -- Use correct type SoftMax_Access_T
      begin 
         Put_Line("  Initializing first Linear layer (Input: 2, Output: 5)..."); 
         Layer1.Initialize(2, 5);  -- No need for type conversion
         My_Model.Add_Layer(Func_Access_T(Layer1)); 
         Put_Line("  First Linear layer added."); 
 
         Put_Line("  Adding ReLU activation layer..."); 
         My_Model.Add_Layer(Func_Access_T(Layer2)); 
         Put_Line("  ReLU layer added."); 
         
         Put_Line("  Initializing second Linear layer (Input: 5, Output: 3)..."); 
         Layer3.Initialize(5, 3);
         My_Model.Add_Layer(Func_Access_T(Layer3)); 
         Put_Line("  Second Linear layer added."); 
         
         Put_Line("  Adding Softmax activation layer..."); 
         My_Model.Add_Layer(Func_Access_T(Layer4)); 
         Put_Line("  Softmax layer added."); 

         -- Add loss function
         Put_Line("  Adding Cross Entropy loss function...");
         My_Model.Add_Loss(Loss_Access_T'(new Cross_Entropy_T));  -- Fixed with qualified expression
         Put_Line("  Loss function added.");
         
         -- Use Create_SGD_T instead of direct aggregate construction
         Put_Line("  Setting SGD optimizer (LR=0.01, Momentum=0.9)...");
         My_Model.Set_Optimizer(Optim_Access_T'(new SGD_T'(
           Create_SGD_T(Learning_Rate => 0.01, Weight_Decay => 0.0, Momentum => 0.9))));
         Put_Line("  Optimizer set.");

         Put_Line("Model Architecture: Linear (2 -> 5) -> ReLU -> Linear (5 -> 3) -> Softmax"); 
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