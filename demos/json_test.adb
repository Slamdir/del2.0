with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;  
with Del; use Del;  
with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Operators; use Del.Operators;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Ada.Directories; use Ada.Directories;

procedure Json_Test is
begin
   Put_Line("Program started");
   Put_Line("=== JSON Loading Test ===");
   
   declare
      My_Model : Del.Model.Model;
      Data_Shape : constant Tensor_Shape_T := (1 => 1, 2 => 2);  -- Single sample, 2 features
      Target_Shape : constant Tensor_Shape_T := (1 => 1, 2 => 4);  -- Single sample, 4 classes
      Json_Filename : constant String := "initial_testing.json";
   begin
      Put_Line("Variables declared");
      
      -- Check if file exists
      if not Exists(Json_Filename) then
         Put_Line("ERROR: Cannot find file: " & Json_Filename);
         Put_Line("Current directory is: " & Current_Directory);
         return;
      else
         Put_Line("Found JSON file: " & Json_Filename);
         Put_Line("File size: " & Ada.Directories.Size(Json_Filename)'Image & " bytes");
      end if;

      Put_Line("Creating model...");
      
      declare
         Layer1 : Func_Access_T := new Linear_T;
      begin
         Put_Line("Initializing layer...");
         Linear_T(Layer1.all).Initialize(2, 4);  
         
         Put_Line("Adding layer to model...");
         Del.Model.Add_Layer(My_Model, Layer1);
         
         Put_Line("Attempting to load and process JSON data...");
         -- Test loading data
         Del.Model.Train_Model_From_JSON(
            Self => My_Model,
            Num_Epochs => 1,
            JSON_File => Json_Filename,
            Data_Shape => Data_Shape,
            Target_Shape => Target_Shape);
            
         Put_Line("JSON loading test completed successfully!");
      end;
   end;
   
exception
   when E : JSON_Parse_Error =>
      Put_Line("Error parsing JSON: " & Exception_Message(E));
      Put_Line("Exception occurred at: " & Exception_Information(E));
   when E : others =>
      Put_Line("Unexpected error: " & Exception_Message(E));
      Put_Line("Exception occurred at: " & Exception_Information(E));
end Json_Test;