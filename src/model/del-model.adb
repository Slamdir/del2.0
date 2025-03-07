with Ada.Containers; use Ada.Containers;
with Ada.Exceptions;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Del.ONNX;
with Del.Operators; use Del.Operators;
with Ada.Numerics.Float_Random;
with Del.Utilities;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Del.Data;

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

   -- Single Training procedure that uses the internal dataset
   procedure Train_Model
     (Self       : in out Model;
      Batch_Size : Positive;
      Num_Epochs : Positive)
   is
      package Util renames Del.Utilities;
      Loss_Value : Element_T;
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

                  -- Reset optimizer internal values for new loop
                  -- COMMENT: Commenting out backprop components
                  -- Self.Optimizer.Zero_Gradient(Self.Layers);

                  -- Feedforward next batch of data
                  Actual_Labels := Self.Run_Layers(Training_Data);

                  -- COMMENT: Commenting out backprop components
                  -- Compute loss
                  -- Loss_Value := Self.Loss_Func.Forward(Training_Labels, Actual_Labels);

                  -- COMMENT: Commenting out backprop components
                  -- Backpropagation
                  -- declare
                  --    Gradient    : Tensor_T := Self.Loss_Func.Backward(Training_Labels, Actual_Labels); 
                  --    Cursor      : Layer_Vectors.Cursor := Self.Layers.Last;
                  -- begin
                  --    while Layer_Vectors.Has_Element(Cursor) loop
                  --       Gradient := Layer_Vectors.Element(Cursor).Backward(Gradient);
                  --       Layer_Vectors.Previous(Cursor);
                  --    end loop;

                  --    -- Apply gradient changes
                  --    Self.Optimizer.Step(Self.Layers);
                  -- end;

                  -- Output progress for user feedback
                  Put_Line("Processed epoch" & epoch'Image & ", batch" & batch'Image);
               end;
            end loop;
         end;
      end loop;
      end;
   end Train_Model;

function Run_Layers (Self : in Model; Input : Tensor_T) return Tensor_T is
begin
   Put_Line("Run_Layers called with input shape: " & 
            Shape(Input)(1)'Image & "," & Shape(Input)(2)'Image);
            
   if Self.Layers.Length = 0 then
      Put_Line("No layers in network");
      return Input;
   end if;

   Put_Line("Network has" & Self.Layers.Length'Image & " layers");

   if Self.Layers.Length = 1 then
      -- Process just the first layer
      declare
         Layer : constant Func_Access_T := Self.Layers.First_Element;
      begin
         Put_Line("Processing single layer");
         return Layer.all.Forward(Input);
      end;
   elsif Self.Layers.Length = 2 then
      -- Process two layers sequentially with explicit variables
      declare
         Layer1 : constant Func_Access_T := Self.Layers.Element(1);
         Layer2 : constant Func_Access_T := Self.Layers.Element(2);
      begin
         Put_Line("Processing layer 1");
         declare
            Output1 : constant Tensor_T := Layer1.all.Forward(Input);
         begin
            Put_Line("Processing layer 2");
            return Layer2.all.Forward(Output1);
         end;
      end;
   else
      -- For more than two layers (uncommon in this case)
      declare
         Current : Tensor_T := Input;
      begin
         for I in 1 .. Integer(Self.Layers.Length) loop
            declare
               Layer : constant Func_Access_T := Self.Layers.Element(I);
            begin
               Put_Line("Processing layer" & I'Image);
               Current := Layer.all.Forward(Current);
            end;
         end loop;
         return Current;
      end;
   end if;
exception
   when E : others =>
      Put_Line("Error in Run_Layers: ");
      Put_Line(Ada.Exceptions.Exception_Information(E));
      raise;
end Run_Layers;

   procedure Export_ONNX(
      Self : in Model;
      Filename : String) is
   begin
      Del.ONNX.Save_ONNX_Model(Self, Filename);
   end Export_ONNX;

end Del.Model;