with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;
with Ada.Containers.Vectors;

with Del;               use Del;
with Del.Model;         use Del.Model;
with Del.Operators;     use Del.Operators;
with Del.Optimizers;    use Del.Optimizers;
with Del.JSON;          use Del.JSON;
with Del.Loss;          use Del.Loss;

with Orka.Numerics.Singles.Tensors;     use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure presentation_four_demo is

   -- Model and Training Parameters
   My_Model      : Del.Model.Model;
   Linear_Layer  : Linear_Access_T;
   ReLU_Layer    : ReLU_Access_T;
   Softmax_Layer : Softmax_Access_T;

   Hidden_Units  : constant Positive := 100;
   Num_Classes   : constant Positive := 3;
   Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);
   Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 3);

   Json_Filename : constant String := "demos/demo-data/spiral_3_5.json";
   Output_File   : constant String := "demos/output/model_output.json";

   Batch_Size    : constant Positive := 20;
   Num_Epochs    : constant Positive := 200;

   Optimizer     : Optim_Access_T := new SGD_T'(Create_SGD_T(
      Learning_Rate => 0.6,
      Weight_Decay  => 0.00001,
      Momentum      => 0.9));

   -- Helper: Print tensor shape and values
   procedure Print_Tensor(T : Tensor_T; Name : String) is
      S : Tensor_Shape_T := Shape(T);
   begin
      Put_Line(Name & " shape: (" & S(1)'Image & ", " & S(2)'Image & ")");
      Put_Line(Name & " values:");
      Put_Line(Image(T));
   end Print_Tensor;

begin
   Put_Line("=============================================================");
   Put_Line("             DEL: Deep Learning in Ada - Demo                ");
   Put_Line("=============================================================");
   New_Line;

   -- Step 1: Model Initialization
   Put_Line("STEP 1: MODEL INITIALIZATION");
   Put_Line("-------------------------------------------------------------");
   Put_Line("Creating model architecture:");
   Put_Line("  - Input Layer: 2 features");
   Put_Line("  - Hidden Layer: " & Hidden_Units'Image & " units + ReLU");
   Put_Line("  - Output Layer: " & Num_Classes'Image & " classes + Softmax");

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

   My_Model.Set_Optimizer(Optimizer);
   My_Model.Add_Loss(new Cross_Entropy_T);
   New_Line;

   -- Step 2: Data Loading
   Put_Line("STEP 2: DATASET CONFIGURATION");
   Put_Line("-------------------------------------------------------------");
   Put_Line("Loading dataset from: " & Json_Filename);
   if not Exists(Json_Filename) then
      Put_Line("ERROR: Dataset file not found.");
      return;
   end if;

   My_Model.Load_Data_From_JSON(
      JSON_File    => Json_Filename,
      Data_Shape   => Data_Shape,
      Target_Shape => Target_Shape);

   Put_Line("Data loaded successfully.");
   Put_Line("Batch Size: " & Batch_Size'Image);
   Put_Line("Epochs: " & Num_Epochs'Image);
   Put_Line("Optimizer: SGD (lr = 0.6, momentum = 0.9, weight decay = 0.00001)");
   New_Line;

   -- Step 3: Training
   Put_Line("STEP 3: MODEL TRAINING");
   Put_Line("-------------------------------------------------------------");
   Put_Line("Beginning training process...");
   My_Model.Train_Model(
      Batch_Size   => Batch_Size,
      Num_Epochs   => Num_Epochs);
   Put_Line("Training complete.");
   New_Line;

   -- Step 4: Export Output
   Put_Line("STEP 4: EXPORT RESULTS");
   Put_Line("-------------------------------------------------------------");
   Put_Line("Exporting model output to file: " & Output_File);
   My_Model.Export_To_JSON(Output_File);
   Put_Line("Export complete. Ready for visualization.");
   New_Line;

   -- Success Footer
   Put_Line("=============================================================");
   Put_Line("              Demo completed successfully!                   ");
   Put_Line("=============================================================");

exception
   when E : JSON_Parse_Error =>
      Put_Line("ERROR: JSON parsing failed.");
      Put_Line("Details: " & Exception_Message(E));
   when E : others =>
      Put_Line("ERROR: An unexpected issue occurred.");
      Put_Line("Details: " & Exception_Message(E));

end presentation_four_demo;
