with Del.Model;
with Del.Operators;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;

procedure Export_Model is
   Network     : Del.Model.Model;
   JSON_File   : constant String := "demos\demo-data\spiral_3.json";
   Output_File : constant String := "demos\output\model_output.json";

   --  layers
   Linear_Layer  : Del.Operators.Linear_Access_T;
   ReLU_Layer    : Del.Operators.ReLU_Access_T;
   Softmax_Layer : Del.Operators.SoftMax_Access_T;

begin
   Put_Line("Initializing model...");

   --  layers to the model
   Linear_Layer := new Del.Operators.Linear_T;
   Linear_Layer.Initialize(2, 4); 
   Network.Add_Layer(Del.Func_Access_T(Linear_Layer));

   ReLU_Layer := new Del.Operators.ReLU_T;
   Network.Add_Layer(Del.Func_Access_T(ReLU_Layer));

   Softmax_Layer := new Del.Operators.SoftMax_T;
   Network.Add_Layer(Del.Func_Access_T(Softmax_Layer));

   -- loading dataset from JSON file
   Put_Line("Loading dataset...");
   Network.Load_Data_From_JSON(
      JSON_File    => JSON_File,
      Data_Shape   => (1 => 1, 2 => 2),  -- Each sample is 1x2
      Target_Shape => (1 => 1, 2 => 4)   
   );
   Put_Line("Dataset successfully loaded.");

   -- train the model
   Put_Line("Starting model training...");
   Network.Train_Model(Batch_Size => 100, Num_Epochs => 100); 
   Put_Line("Training complete.");

   Put_Line("Exporting model output to JSON...");
   Network.Export_To_JSON(Output_File);
   Put_Line("Export complete. Output written to: " & Output_File);
   
exception
   when E : others =>
      Put_Line("Error in Export_Model: " & Ada.Exceptions.Exception_Message(E));
end Export_Model;