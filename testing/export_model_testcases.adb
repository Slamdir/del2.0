with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;

with Del; use Del;
with Del.Model;
with Del.Operators;
with Del.Optimizers;
with Del.Loss;
with Del.JSON;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Export_Model_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;
   package DOpt renames Del.Optimizers;
   package DL renames Del.Loss;

   Json_Filename  : constant String := "demos/demo-data/spiral_3_6.json";
   Output_Filename : constant String := "demos/output/model_output.json";

   Input_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);
   Target_Shape   : constant Tensor_Shape_T := (1 => 1, 2 => 3);

   Batch_Size     : constant Positive := 15;
   Num_Epochs     : constant Positive := 20;

   Hidden_Units   : constant Positive := 50;
   Num_Classes    : constant Positive := 3;

   My_Model : DMod.Model;
   Optimizer : DOpt.SGD_T;  -- Just a simple SGD_T object

   -- Assertion procedure
   procedure Assert_Condition(Condition : Boolean; Message : String) is
   begin
      if Condition then
         Put_Line(Message & " Passed");
      else
         Put_Line(Message & " Failed");
      end if;
   end Assert_Condition;

begin
   Put_Line("=== Export Model Testcases ===");

   -- Step 1: Build model
   declare
      Linear1 : D.Func_Access_T := new DOp.Linear_T;
      ReLU1 : D.Func_Access_T := new DOp.ReLU_T;
      Linear2 : D.Func_Access_T := new DOp.Linear_T;
      Softmax1 : D.Func_Access_T := new DOp.SoftMax_T;
   begin
      -- Initialize layers
      DOp.Linear_T(Linear1.all).Initialize(2, Hidden_Units);
      DOp.Linear_T(Linear2.all).Initialize(Hidden_Units, Num_Classes);

      -- Add to model
      My_Model.Add_Layer(Linear1);
      My_Model.Add_Layer(ReLU1);
      My_Model.Add_Layer(Linear2);
      My_Model.Add_Layer(Softmax1);
   end;

   -- Set optimizer and loss
   My_Model.Set_Optimizer(new DOpt.SGD_T);
   My_Model.Add_Loss(new DL.Cross_Entropy_T);

   -- Step 2: Load dataset
   if not Exists(Json_Filename) then
      Put_Line("ERROR: JSON input file not found: " & Json_Filename);
      return;
   end if;

   begin
      My_Model.Load_Data_From_JSON(
         JSON_File    => Json_Filename,
         Data_Shape   => Input_Shape,
         Target_Shape => Target_Shape);

      Put_Line("Data loaded successfully.");
   exception
      when others =>
         Put_Line("ERROR: Unexpected error while loading JSON.");
         return;
   end;

   -- Step 3: Train Model
   begin
      Put_Line("Training model...");
      My_Model.Train_Model(
         Batch_Size => Batch_Size,
         Num_Epochs => Num_Epochs);

      Put_Line("Training completed successfully.");
   exception
      when others =>
         Put_Line("ERROR: Unexpected error during training.");
         return;
   end;

   -- Step 4: Export Model Output
   begin
      Put_Line("Exporting model output...");
      My_Model.Export_To_JSON(Output_Filename);

      Assert_Condition(Exists(Output_Filename), "Model Exported to JSON");
   exception
      when others =>
         Put_Line("ERROR: Unexpected error during export.");
         Assert_Condition(False, "Model Exported to JSON");
   end;

   Put_Line("=== Export Model Testcases Completed ===");
end Export_Model_Testcases;
