-- File: del-json.adb
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Containers; use Ada.Containers;
with GNATCOLL.JSON; use GNATCOLL.JSON;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Del; use Del;

package body Del.JSON is
function To_Element(Value : JSON_Value) return Element_T is
   begin
      if Kind(Value) = JSON_Float_Type then
         return Element_T(Float'(Value.Get)); 
      elsif Kind(Value) = JSON_Int_Type then
         return Element_T(Float(Integer'(Value.Get)));  
      else
         raise JSON_Parse_Error with "Invalid numeric type in JSON";
      end if;
   end To_Element;
   
   procedure Validate_Dimensions
     (JSON_Data : JSON_Array;
      Shape     : Tensor_Shape_T) is
      
      Expected_Size : Positive := 1;
   begin
      -- Calculate expected total size from shape
      for Dim of Shape loop
         Expected_Size := Expected_Size * Dim;
      end loop;
      
      -- Check if JSON array size matches expected size
      if Length(JSON_Data) /= Expected_Size then
         raise JSON_Parse_Error with
           "JSON array size" & Length(JSON_Data)'Image &
           " does not match tensor shape size" & Expected_Size'Image;
      end if;
   end Validate_Dimensions;
   
   function From_JSON_Array
     (JSON_Data : JSON_Array;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
      Result : Tensor_T := Zeros(Shape);
      Index  : Natural := 1;
   begin
      -- Validate dimensions before processing
      Validate_Dimensions(JSON_Data, Shape);
      
      -- Convert flat JSON array to tensor based on shape
      for I in 1 .. Length(JSON_Data) loop
         declare
            Value : constant JSON_Value := Get(JSON_Data, I);
            Element : constant Element_T := To_Element(Value);
            -- Calculate multi-dimensional indices based on current position
            Row : constant Positive := (Index - 1) / Shape(2) + 1;
            Col : constant Positive := (Index - 1) mod Shape(2) + 1;
         begin
            Result.Set((Row, Col), Element);
            Index := Index + 1;
         end;
      end loop;
      
      return Result;
   end From_JSON_Array;
   
   function Get_JSON_Array
     (Object : JSON_Value;
      Field  : String) return JSON_Array is
   begin
      if Kind(Object) /= JSON_Object_Type then
         raise JSON_Parse_Error with "Expected JSON object";
      end if;
      
      declare
         Value : constant JSON_Value := Get(Object, Field);
      begin
         if Kind(Value) /= JSON_Array_Type then
            raise JSON_Parse_Error with
              "Field '" & Field & "' must be an array";
         end if;
         return Get(Value);
      end;
   end Get_JSON_Array;
   
   function Load_JSON_Tensor
     (Filename : String;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
      File    : File_Type;
      Content : Unbounded_String;
   begin
      -- Read JSON file
      Open(File, In_File, Filename);
      while not End_Of_File(File) loop
         Append(Content, Get_Line(File));
      end loop;
      Close(File);
      
      -- Parse JSON and convert to tensor
      return Parse_JSON_Tensor(To_String(Content), Shape);
   exception
      when E : others =>
         if Is_Open(File) then
            Close(File);
         end if;
         raise JSON_Parse_Error with
           "Error loading JSON file: " & Ada.Exceptions.Exception_Message(E);
   end Load_JSON_Tensor;
   
   function Parse_JSON_Tensor
     (JSON_Str : String;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
      Data : constant JSON_Value := Read(JSON_Str);
   begin
      if Kind(Data) /= JSON_Array_Type then
         raise JSON_Parse_Error with "JSON data must be an array";
      end if;
      
      return From_JSON_Array(Get(Data), Shape);
   exception
      when E : others =>
         raise JSON_Parse_Error with
           "Error parsing JSON string: " & Ada.Exceptions.Exception_Message(E);
   end Parse_JSON_Tensor;
   
function Load_Dataset
     (Filename     : String;
      Data_Shape   : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T) return Dataset_Array
   is
      File    : File_Type;
      Content : Unbounded_String;
   begin
      -- Read JSON file
      Open(File, In_File, Filename);
      while not End_Of_File(File) loop
         Append(Content, Get_Line(File));
      end loop;
      Close(File);
      
      declare
         JSON_Data : constant JSON_Value := Read(To_String(Content));
         Data_Array : constant JSON_Array := Get_JSON_Array(JSON_Data, "data");
         Label_Array : constant JSON_Array := Get_JSON_Array(JSON_Data, "labels");
         Dataset : Dataset_Array(1 .. 1);  -- Start with one entry
      begin
         Put_Line("Data array length:" & Length(Data_Array)'Image);
         Put_Line("Label array length:" & Length(Label_Array)'Image);
         
         -- Convert the first data entry into a tensor
         declare
            First_Data : constant JSON_Array := Get(Get(Data_Array, 1));
         begin
            Dataset(1).Data := new Tensor_T'(From_JSON_Array(First_Data, Data_Shape));
         end;

         
         declare
            Label_Value : constant JSON_Value := Get(Label_Array, 1);
            Label : constant Integer := Integer'(Label_Value.Get);
            Target : Tensor_T := Zeros(Target_Shape);
         begin
            Put_Line("Label value:" & Label'Image);
            
            Target.Set((1, Label), 1.0);
            Dataset(1).Target := new Tensor_T'(Target);
         end;
         
         return Dataset;
      end;
   exception
      when E : others =>
         if Is_Open(File) then
            Close(File);
         end if;
         Put_Line("Error details: " & Ada.Exceptions.Exception_Information(E));
         raise JSON_Parse_Error with
           "Error loading dataset: " & Ada.Exceptions.Exception_Message(E);
   end Load_Dataset;
   
end Del.JSON;