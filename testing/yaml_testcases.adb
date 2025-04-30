with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Directories; use Ada.Directories;

with Del; use Del;
with Del.Model;
with Del.Data; use Del.Data;
with Del.YAML; use Del.YAML;
with Del.Operators; use Del.Operators;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure YAML_Testcases is
   package D renames Del;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Constants
   YAML_Filename : constant String := "demos/demo-data/initial_testing.yaml";
   Data_Shape    : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- 1 sample, 2 features
   Target_Shape  : constant Tensor_Shape_T := (1 => 1, 2 => 4);  -- 1 sample, 4 classes
   Batch_Size    : constant Positive := 10;
   Num_Epochs    : constant Positive := 1;

   -- Model
   My_Model : DMod.Model;

   -- Simple Assertion
   procedure Assert_Condition(Condition : Boolean; Message : String) is
   begin
      if Condition then
         Put_Line(Message & " Passed");
      else
         Put_Line(Message & " Failed");
      end if;
   end Assert_Condition;

begin
   Put_Line("=== YAML Loading and Training Testcases ===");

   -- 1. Check YAML file exists
   Put_Line("Checking YAML file existence...");
   Assert_Condition(Exists(YAML_Filename), "YAML File Exists");

   if not Exists(YAML_Filename) then
      Put_Line("ERROR: YAML file missing, aborting tests.");
      return;
   end if;

   -- 2. Build Model
   declare
      Layer1 : Func_Access_T := new Linear_T;
      Layer2 : Func_Access_T := new ReLU_T;
   begin
      Linear_T(Layer1.all).Initialize(2, 5);
      My_Model.Add_Layer(Layer1);
      My_Model.Add_Layer(Layer2);
   end;

   Put_Line("Model layers added successfully.");

   -- 3. Load Data
   begin
      Put_Line("Loading data from YAML...");
      My_Model.Load_Data_From_YAML(
         YAML_File     => YAML_Filename,
         Data_Shape    => Data_Shape,
         Target_Shape  => Target_Shape);

      Put_Line("Data loading from YAML Passed");
   exception
      when YAML_Parse_Error =>
         Put_Line("ERROR: YAML parsing failed.");
         return;
      when others =>
         Put_Line("ERROR: Unexpected error during YAML loading.");
         return;
   end;

   -- 4. Train Model
   begin
      Put_Line("Starting model training...");
      My_Model.Train_Model(
         Batch_Size => Batch_Size,
         Num_Epochs => Num_Epochs);

      Put_Line("Training Completed Successfully.");
      Assert_Condition(True, "Model Training Test");
   exception
      when others =>
         Put_Line("ERROR: Unexpected error during training.");
         Assert_Condition(False, "Model Training Test");
   end;

   Put_Line("=== YAML Testcases Completed ===");
end YAML_Testcases;
