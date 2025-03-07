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
   
   function Load_From_JSON
     (JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access
   is
      Dataset : constant Dataset_Array := Load_Dataset
        (Filename     => JSON_File,
         Data_Shape   => Data_Shape,
         Target_Shape => Target_Shape);
   begin
      Put_Line("Loading data from JSON file: " & JSON_File);
      Put_Line("Dataset loaded successfully. Samples:" & Dataset'Length'Image);
      return new Training_Data'
        (Data   => new Tensor_T'(Dataset(1).Data.all),
         Labels => new Tensor_T'(Dataset(1).Target.all));
   exception
      when E : JSON_Parse_Error =>
         Put_Line("Error loading JSON data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Load_From_JSON;
end Del.Data;