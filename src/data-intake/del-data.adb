with Ada.Exceptions;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Deallocation;

package body Del.Data is
   function Create
     (Data   : Tensor_T;
      Labels : Tensor_T) return Training_Data_Access
   is
   begin
      return new Training_Data'
        (Data   => new Tensor_T'(Data),
         Labels => new Tensor_T'(Labels));
   end Create;
   
   function Get_Data(Self : Training_Data) return Tensor_T is
   begin
      return Self.Data.all;
   end Get_Data;
   
   function Get_Labels(Self : Training_Data) return Tensor_T is
   begin
      return Self.Labels.all;
   end Get_Labels;

function Combine_Dataset_Samples
  (Dataset     : Dataset_Array;
   Data_Shape  : Tensor_Shape_T;
   Label_Shape : Tensor_Shape_T) return Training_Data_Access
is
   -- Calculate combined dimensions
   Sample_Count : constant Positive := Dataset'Length;
   
   -- Create empty tensors for the combined data
   -- First create with the correct dimensions
   Combined_Data : Tensor_T := Zeros((Sample_Count, Data_Shape(2)));
   Combined_Labels : Tensor_T := Zeros((Sample_Count, Label_Shape(2)));
begin
   Put_Line("Combining" & Sample_Count'Image & " samples into training dataset");
   
   -- Copy each sample's data into the combined tensors using a different approach
   for I in Dataset'Range loop
      -- Copy sample data and labels row by row
      declare
         Sample_Data_Row : Tensor_T := Dataset(I).Data.all;
         Sample_Label_Row : Tensor_T := Dataset(I).Target.all;
      begin
         -- Use index to set entire row at once
         Combined_Data.Set(Index => I, Value => Sample_Data_Row);
         Combined_Labels.Set(Index => I, Value => Sample_Label_Row);
      end;
   end loop;
   
   -- Return the combined dataset
   return new Training_Data'
     (Data   => new Tensor_T'(Combined_Data),
      Labels => new Tensor_T'(Combined_Labels));
end Combine_Dataset_Samples;
   
   function Load_From_JSON
  (JSON_File     : String;
   Data_Shape    : Tensor_Shape_T;
   Target_Shape  : Tensor_Shape_T) return Training_Data_Access
is
   Dataset : constant Dataset_Array := Del.JSON.Load_Dataset
     (Filename     => JSON_File,
      Data_Shape   => Data_Shape,
      Target_Shape => Target_Shape);
begin
   Put_Line("Loading data from JSON file: " & JSON_File);
   Put_Line("Dataset loaded successfully. Samples:" & Dataset'Length'Image);
   
   -- Use the new function to combine all samples
   return Combine_Dataset_Samples(Dataset, Data_Shape, Target_Shape);
end Load_From_JSON;
end Del.Data;