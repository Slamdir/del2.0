with Ada.Containers; use Ada.Containers;
with Ada.Exceptions;
with Del.Loss;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Del.ONNX;
with Del.Operators; use Del.Operators;
with Ada.Numerics.Float_Random;
with Del.Utilities;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Del.Data;
with Del.YAML; use Del.YAML;
with Del.Export; 
package body Del.Model is
   procedure Add_Layer (Self : in out Model; Layer : Func_Access_T) is
   begin
      Self.Layers.Append (Layer);
   end Add_Layer;

   procedure Add_Loss (Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

   function Get_Layers_Vector(Self : Model) return Layer_Vectors.Vector is
   begin
      return Self.Layers;
   end Get_Layers_Vector;

   procedure Set_Optimizer(Self : in out Model; Optimizer : Optim_Access_T) is
   begin
      Self.Optimizer := Optimizer;
   end Set_Optimizer;
   
   -- Data management procedures
   procedure Set_Dataset(Self : in out Model; Dataset : Training_Data_Access) is
      procedure Free is new Ada.Unchecked_Deallocation
        (Object => Del.Data.Training_Data'Class, Name => Del.Data.Training_Data_Access);
   begin
      -- Free any existing dataset
      if Self.Dataset /= null then
         Free(Self.Dataset);
      end if;
      
      -- Store the provided dataset
      Self.Dataset := Dataset;
   end Set_Dataset;
   
   function Get_Dataset(Self : Model) return Training_Data_Access is
   begin
      return Self.Dataset;
   end Get_Dataset;
   
   procedure Load_Data_From_JSON
     (Self          : in out Model;
      JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) 
   is
   begin
      Put_Line("Loading data from JSON file: " & JSON_File);
      
      -- Use the Del.Data package to load the dataset
      Set_Dataset(
         Self    => Self,
         Dataset => Del.Data.Load_From_JSON(
            JSON_File    => JSON_File,
            Data_Shape   => Data_Shape,
            Target_Shape => Target_Shape));
            
      Put_Line("Dataset loaded successfully");
   exception
      when E : JSON_Parse_Error =>
         Put_Line("Error loading JSON data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Load_Data_From_JSON;

   function Do_Backward(S : Model; C : Layer_Vectors.Cursor; IT : Tensor_T) return Tensor_T is
      use Layer_Vectors;
      T : Tensor_T := Layer_Vectors.Element(C).Backward (IT);
   begin
      if C = S.Layers.First then
         return T;
      else
         return Do_Backward (S, Layer_Vectors.Previous(C), T);
      end if;
   end;

   -- Single Training procedure that uses the internal dataset
   procedure Train_Model
     (Self       : in out Model;
      Batch_Size : Positive;
      Num_Epochs : Positive)
   is
      package Util renames Del.Utilities;
      Loss_Value : Float;
   begin
      -- Check if dataset is loaded
      if Self.Dataset = null then
         raise Program_Error with "No dataset loaded for training";
      end if;
      
      -- Get data and labels from the dataset
      declare
         Data   : constant Tensor_T := Self.Dataset.Get_Data;
         Labels : constant Tensor_T := Self.Dataset.Get_Labels;
      begin
         Put_Line("Starting training with:" & 
                 " Data shape: (" & Shape(Data)(1)'Image & "," & Shape(Data)(2)'Image & ")" &
                 " Labels shape: (" & Shape(Labels)(1)'Image & "," & Shape(Labels)(2)'Image & ")");

      for epoch in 1 .. Num_Epochs loop
         declare
            -- Shuffle indices
            Indices : Util.Integer_Array := Util.Generate_Random_List(Shape(Data)(1));
         begin
            -- Loop across number of batches in data (last one may be incomplete)
            for batch in 1 .. (Shape(Data)(1) / Batch_Size) loop
               declare
                  Training_Data   : Tensor_T := Zeros((Batch_Size, Shape(Data)(2)));
                  Training_Labels : Tensor_T := Zeros((Batch_Size, Shape(Labels)(2)));
                  Actual_Labels   : Tensor_T := Zeros((Batch_Size, Shape(Labels)(2)));
                  max_Index       : Positive;
                  data_Index      : Positive := 1;
               begin
                  if Shape(Data)(1) < (batch * Batch_Size) then
                     max_Index := Shape(Data)(1);
                  else
                     max_Index := (batch * Batch_Size);
                  end if;

                  -- Grab batch of training data and labels
                  for batch_Index in (((batch - 1) * Batch_Size) + 1) .. max_Index loop
                     declare
                        Row_Data  : Tensor_T := Data(Indices(batch_Index));
                        Row_Label : Tensor_T := Labels(Indices(batch_Index));
                     begin
                        Training_Data.Set(Index => data_Index, Value => Row_Data);
                        Training_Labels.Set(Index => data_Index, Value => Row_Label);
                        data_Index := data_Index + 1;
                     end;
                  end loop;

                  Put_Line("Before Zero Grad");
                  -- Reset optimizer internal values for new loop
                  Self.Optimizer.Zero_Gradient(Self.Layers);
                  Put_Line("After Zero Grad");
                  New_Line;
                  
                  Put_Line("Before Run Layers");
                  -- Feedforward next batch of data
                  Actual_Labels := Self.Run_Layers(Training_Data);
                  Put_Line("After Run Layers");
                  New_Line;

                  Put_Line("Before Compute Loss");
                  Put_Line("Target Labels " & Training_Labels.Image);
                  New_Line;

                  Put_Line("Actual Labels " & Actual_Labels.Image);
                  New_Line;

                  -- Compute loss
                  Loss_Value := Self.Loss_Func.Forward(Training_Labels, Actual_Labels);
                  Put_Line("After Compute Loss " & Loss_Value'Image);
                  New_Line;

                  -- Backpropagation
                  declare
                     Gradient    : Tensor_T := Self.Loss_Func.Backward (Training_Labels, Actual_Labels); 
                     Cursor      : Layer_Vectors.Cursor := Self.Layers.Last;
                     New_Grad    : Tensor_T := Do_Backward (Self, Cursor, Gradient);
                  begin
                     -- Apply gradient changes
                     Self.Optimizer.Step (Self.Layers);
                  end;

                  -- Output progress for user feedback
                  Put_Line("Processed epoch" & epoch'Image & ", batch" & batch'Image);
               end;
            end loop;
         end;
      end loop;
      end;
   end Train_Model;

   function Do_Forward (S : Model; C : Layer_Vectors.Cursor; IT : Tensor_T) return Tensor_T is
      use Layer_Vectors;
      T : Tensor_T := Layer_Vectors.Element (C).Forward (IT);
   begin
      if C = S.Layers.Last then
         return T;
      else
         return Do_Forward (S, Layer_Vectors.Next (C), T);
      end if;
   end;

   function Run_Layers (Self : in Model; Input : Tensor_T) return Tensor_T is
      C : Layer_Vectors.Cursor := Self.Layers.First;
   begin
      Put_Line ("Run_Layers called with input shape: " & 
                Shape (Input) (1)'Image & "," & Shape (Input) (2)'Image);
               
      if Self.Layers.Length = 0 then
         Put_Line ("No layers in network");
         return Input;
      end if;

   Put_Line("Network has" & Self.Layers.Length'Image & " layers");

      return Do_Forward (Self, C, Input);
   exception
      when E : others =>
         Put_Line ("Error in Run_Layers: ");
         Put_Line (Ada.Exceptions.Exception_Information (E));
         raise;
   end Run_Layers;

   -- Add these new procedures to del-model.adb

   procedure Load_Data_From_YAML
     (Self          : in out Model;
      YAML_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) 
   is
   begin
      Put_Line("Loading data from YAML file: " & YAML_File);
      
      -- Use the Del.Data package to load the dataset
      Set_Dataset(
         Self    => Self,
         Dataset => Del.Data.Load_From_YAML(
            YAML_File    => YAML_File,
            Data_Shape   => Data_Shape,
            Target_Shape => Target_Shape));
            
      Put_Line("YAML dataset loaded successfully");
   exception
      when E : YAML_Parse_Error =>
         Put_Line("Error loading YAML data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Load_Data_From_YAML;

   procedure Export_To_JSON(Self : in Model; Filename : String) is
   begin
      Del.Export.Export_To_JSON(Self, Filename);
   end Export_To_JSON;

   procedure Load_Data_From_File
     (Self          : in out Model;
      Filename      : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T)
   is
   begin
      Put_Line("Loading data from file: " & Filename);
      
      -- Use the Del.Data package to automatically detect and load the dataset
      Set_Dataset(
         Self    => Self,
         Dataset => Del.Data.Load_From_File(
            Filename     => Filename,
            Data_Shape   => Data_Shape,
            Target_Shape => Target_Shape));
            
      Put_Line("Dataset loaded successfully");
   exception
      when E : others =>
         Put_Line("Error loading data: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Load_Data_From_File;


   procedure Export_ONNX(
      Self : in Model;
      Filename : String) is
   begin
      Del.ONNX.Save_ONNX_Model(Self, Filename);
   end Export_ONNX;

end Del.Model;