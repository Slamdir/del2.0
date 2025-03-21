with Ada.Exceptions;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Deallocation;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

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
      Dataset : constant Dataset_Array := Del.JSON.Load_Dataset
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
   
   function Load_From_YAML
     (YAML_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access
   is
      Dataset : constant Dataset_Array := Del.YAML.Load_Dataset
        (Filename     => YAML_File,
         Data_Shape   => Data_Shape,
         Target_Shape => Target_Shape);
   begin
      Put_Line("Loading data from YAML file: " & YAML_File);
      Put_Line("Dataset loaded successfully. Samples:" & Dataset'Length'Image);
      return new Training_Data'
        (Data   => new Tensor_T'(Dataset(1).Data.all),
         Labels => new Tensor_T'(Dataset(1).Target.all));
   exception
      when E : YAML_Parse_Error =>
         Put_Line("Error loading YAML data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Load_From_YAML;
   
   function Load_From_File
     (Filename      : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access
   is
      Ext_Start : Natural := Index(Filename, ".", Ada.Strings.Backward);
      Extension : String := Filename(Ext_Start+1..Filename'Last);
   begin
      Put_Line("Detected file extension: " & Extension);
      
      -- Choose loading method based on file extension
      if Extension = "json" then
         return Load_From_JSON(Filename, Data_Shape, Target_Shape);
      elsif Extension = "yaml" or Extension = "yml" then
         return Load_From_YAML(Filename, Data_Shape, Target_Shape);
      else
         raise Constraint_Error with 
           "Unsupported file extension: " & Extension & ". Supported types: .json, .yaml, .yml";
      end if;
   end Load_From_File;
   
end Del.Data;