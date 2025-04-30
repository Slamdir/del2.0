with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;
with Del; use Del;
with Del.Model; use Del.Model;
with Del.Operators; use Del.Operators;
with Del.Optimizers; use Del.Optimizers;
with Del.Loss; use Del.Loss;
with Del.JSON; use Del.JSON;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Export_Model is
   -- Model and data declarations
   My_Model      : Del.Model.Model;
   Linear1       : Linear_Access_T;
   Linear2       : Linear_Access_T;
   Softmax_Layer : Softmax_Access_T;
   Optimizer     : Optim_Access_T;
   Input_Shape   : constant Tensor_Shape_T := (1 => 1, 2 => 2); -- 2 features
   Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 3); -- 3 classes
   Batch_Size    : constant Positive := 30;
   Num_Epochs    : constant Positive := 50;
   Json_Filename : constant String := "demos/demo-data/spiral_3_5.json";
   Output_File   : constant String := "demos/output/model_output.json";

begin
   -- Step 1: Build the model
   Put_Line("Building model...");
   
   Linear1 := new Linear_T;
   Linear1.Initialize(2, 30);
   My_Model.Add_Layer(Del.Func_Access_T(Linear1));

   Linear2 := new Linear_T;
   Linear2.Initialize(30, 3);
   My_Model.Add_Layer(Del.Func_Access_T(Linear2));

   Softmax_Layer := new Softmax_T;
   My_Model.Add_Layer(Del.Func_Access_T(Softmax_Layer));

   -- Set Optimizer
   Optimizer := new SGD_T'(Create_SGD_T(
      Learning_Rate => 1.0,
      Weight_Decay  => 0.001,
      Momentum      => 0.9));
   My_Model.Set_Optimizer(Optimizer);

   -- Set Loss
   My_Model.Add_Loss(new Cross_Entropy_T);

   New_Line;

   -- Step 2: Load Dataset
   Put_Line("Loading dataset from JSON: " & Json_Filename);
   if not Exists(Json_Filename) then
      Put_Line("ERROR: JSON file not found: " & Json_Filename);
      return;
   end if;

   My_Model.Load_Data_From_JSON(
      JSON_File   => Json_Filename,
      Data_Shape  => Input_Shape,
      Target_Shape => Target_Shape);

   New_Line;

   -- Step 3: Train the Model
   Put_Line("Training model...");
   My_Model.Train_Model(
      Batch_Size => Batch_Size,
      Num_Epochs => Num_Epochs);

   New_Line;

   -- Step 4: Export trained model output
   Put_Line("Exporting trained model output to JSON...");
   My_Model.Export_To_JSON(Output_File);
   Put_Line("Export complete! File saved to: " & Output_File);

   New_Line;
   Put_Line("===========================================================");
   Put_Line("       Model training and export completed successfully!   ");
   Put_Line("===========================================================");

exception
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed - " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: Unexpected issue - " & Exception_Message(E));
end Export_Model;