with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Containers; use Ada.Containers;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Del; use Del;

package body Del.YAML is
function Parse_Coordinate_Data(YAML_Content : String) return Elements_T is
   Result     : Elements_T(1..1000);  -- Adjust size as needed
   Item_Count : Natural := 0;
   Start_Pos  : Natural := YAML_Content'First;
   Line_End   : Natural;
begin
   while Start_Pos <= YAML_Content'Last loop
      -- Find the end of the current line
      Line_End := Index(YAML_Content(Start_Pos..YAML_Content'Last), "" & ASCII.LF);
      if Line_End = 0 then
         Line_End := YAML_Content'Last;
      else
         Line_End := Line_End - 1; -- Exclude the newline
      end if;
      
      -- Extract the current line
      declare
         Line : constant String := Trim(YAML_Content(Start_Pos..Line_End), Ada.Strings.Both);
         Open_Bracket_Pos, Close_Bracket_Pos, Comma_Pos : Natural;
      begin
         if Line'Length > 0 and then Index(Line, "[") > 0 then
            Open_Bracket_Pos := Index(Line, "[");
            Close_Bracket_Pos := Index(Line, "]");
            Comma_Pos := Index(Line(Open_Bracket_Pos..Close_Bracket_Pos), ",");
            
            if Open_Bracket_Pos > 0 and then Comma_Pos > 0 and then Close_Bracket_Pos > 0 and then
               Open_Bracket_Pos < Comma_Pos and then Comma_Pos < Close_Bracket_Pos then
               declare
                  X_Str : constant String := Trim(Line(Open_Bracket_Pos+1..Comma_Pos-1), Ada.Strings.Both);
                  Y_Str : constant String := Trim(Line(Comma_Pos+1..Close_Bracket_Pos-1), Ada.Strings.Both);
                  X_Val, Y_Val : Element_T;
               begin
                  --Put_Line("Debug: Line = '" & Line & "', X_Str = '" & X_Str & "', Y_Str = '" & Y_Str & "'");
                  
                  if X_Str'Length > 0 and Y_Str'Length > 0 then
                     begin
                        X_Val := Element_T'Value(X_Str);
                        Y_Val := Element_T'Value(Y_Str);
                        
                        Item_Count := Item_Count + 1;
                        Result(Item_Count) := X_Val;
                        Item_Count := Item_Count + 1;
                        Result(Item_Count) := Y_Val;
                     exception
                        when Constraint_Error =>
                           Put_Line("Error: Invalid numeric value - X_Str='" & X_Str & "', Y_Str='" & Y_Str & "'");
                           raise YAML_Parse_Error with "Invalid numeric value in line: '" & Line & "'";
                     end;
                  end if;
               end;
            end if;
         end if;
      end;
      
      -- Move to the next line
      Start_Pos := Line_End + 2; -- Skip the newline character
   end loop;
   
   return Result(1..Item_Count);
exception
   when E : others =>
      Put_Line("Error parsing coordinates at position" & Start_Pos'Image & ": " & Exception_Message(E));
      raise YAML_Parse_Error with "Error parsing coordinates: " & Exception_Message(E);
end Parse_Coordinate_Data;

   -- Direct parser for label data format in YAML
function Parse_Label_Data(YAML_Content : String) return Elements_T is
   Result     : Elements_T(1..500);
   Item_Count : Natural := 0;
   Start_Pos  : Natural := YAML_Content'First;
   Line_End   : Natural;
begin
   --Put_Line("Debug: Starting Parse_Label_Data with YAML_Content length =" & YAML_Content'Length'Image);
   
   while Start_Pos <= YAML_Content'Last loop
      -- Find end of current line
      Line_End := Index(Source => YAML_Content(Start_Pos..YAML_Content'Last), 
                         Pattern => ASCII.LF & "",  -- Convert to string
                         From => Start_Pos);
                         
      if Line_End = 0 then
         Line_End := YAML_Content'Last;
      else
         Line_End := Line_End - 1; -- Exclude newline
      end if;
      
      -- Process line if it starts with "  - " or "- "
      declare
         Line : constant String := YAML_Content(Start_Pos..Line_End);
      begin
         -- Check if this is a label entry
         if (Line'Length > 3 and then Line(Line'First..Line'First+2) = "  -") or
            (Line'Length > 1 and then Line(Line'First..Line'First) = "-") then
            -- Extract numeric part
            declare
               Dash_Pos : Natural := Index(Source => Line, 
                                          Pattern => "-",
                                          From => Line'First);
               Value_Start : Natural;
            begin
               if Dash_Pos > 0 then
                  Value_Start := Dash_Pos + 1;
                  -- Skip any spaces after the dash
                  while Value_Start <= Line'Last and then Line(Value_Start) = ' ' loop
                     Value_Start := Value_Start + 1;
                  end loop;
                  
                  if Value_Start <= Line'Last then
                     declare
                        Label_Str : String := Trim(Line(Value_Start..Line'Last), Ada.Strings.Both);
                     begin
                        if Label_Str'Length > 0 then
                           -- Convert to Element_T
                           Item_Count := Item_Count + 1;
                           Result(Item_Count) := Element_T'Value(Label_Str);
                           --Put_Line("Debug: Label #" & Item_Count'Image & " = " & Label_Str);
                        end if;
                     end;
                  end if;
               end if;
            end;
         end if;
      end;
      
      -- Move to next line
      Start_Pos := Line_End + 2;
      if Start_Pos > YAML_Content'Last then
         exit;
      end if;
   end loop;
   
   --Put_Line("Debug: Total labels parsed =" & Item_Count'Image);
   return Result(1..Item_Count);
exception
   when E : others =>
      Put_Line("Error parsing label at Item_Count =" & Item_Count'Image & ": " & 
               Exception_Message(E));
      raise YAML_Parse_Error with "Error parsing labels: " & Exception_Message(E);
end Parse_Label_Data;

   function To_Element(Value : String) return Element_T is
      Trimmed_Value : String := Trim(Value, Ada.Strings.Both);
   begin
      -- Direct numeric value
      return Element_T'Value(Trimmed_Value);
   exception
      when E : others =>
         raise YAML_Parse_Error with "Invalid numeric value in YAML: '" & Trimmed_Value & "' - " & 
                                    Exception_Message(E);
   end To_Element;
   
   function To_Element_Array(Value : String) return Elements_T is
      Trimmed_Value : String := Trim(Value, Ada.Strings.Both);
      Result : Elements_T(1..2); -- Assuming each data point has 2 elements
   begin
      -- Check if the value is in array notation [x, y]
      if Trimmed_Value(Trimmed_Value'First) = '[' and then 
         Trimmed_Value(Trimmed_Value'Last) = ']' then
         -- Extract content between brackets
         declare
            Content : String := Trimmed_Value(Trimmed_Value'First+1 .. Trimmed_Value'Last-1);
            Comma_Pos : Natural := Index(Content, ",");
            First_Value : String := Trim(Content(Content'First .. Comma_Pos-1), Ada.Strings.Both);
            Second_Value : String := Trim(Content(Comma_Pos+1 .. Content'Last), Ada.Strings.Both);
         begin
            Result(1) := Element_T'Value(First_Value);
            Result(2) := Element_T'Value(Second_Value);
            return Result;
         end;
      else
         -- Not in array format, handle error
         raise YAML_Parse_Error with "Expected array format [x, y] but got: '" & Trimmed_Value & "'";
      end if;
   exception
      when E : others =>
         declare
            Msg : constant String := Exception_Message(E);
         begin
            if Msg'Length > 0 then
               raise YAML_Parse_Error with "Invalid array value in YAML: '" & Trimmed_Value & "' - " & Msg;
            else
               raise YAML_Parse_Error with "Invalid array value in YAML: '" & Trimmed_Value & "'";
            end if;
         end;
   end To_Element_Array;
   
   procedure Validate_Dimensions
     (YAML_Data : String;
      Shape     : Tensor_Shape_T) is
      
      Expected_Size : Positive := 1;
      Item_Count    : Natural := 0;
      Start_Pos     : Natural := YAML_Data'First;
      Dash_Pos      : Natural;
   begin
      -- Calculate expected total size from shape
      for Dim of Shape loop
         Expected_Size := Expected_Size * Dim;
      end loop;
      
      -- Count items in YAML array (counting dashes at start of lines)
      while Start_Pos <= YAML_Data'Last loop
         Dash_Pos := Index(YAML_Data(Start_Pos..YAML_Data'Last), "- ");
         exit when Dash_Pos = 0;
         
         -- For data items that are arrays like [x, y], each array counts as two items
         if Dash_Pos+2 <= YAML_Data'Last and then 
            Index(YAML_Data(Dash_Pos+2..YAML_Data'Last), "[") = Dash_Pos+2 then
            -- This is an array notation, count as 2 elements
            Item_Count := Item_Count + 2;
         else
            -- Regular item
            Item_Count := Item_Count + 1;
         end if;
         
         Start_Pos := Dash_Pos + 2;
      end loop;
      
      -- Check if YAML array size matches expected size
      if Item_Count /= Expected_Size then
         Put_Line("Warning: YAML array size" & Item_Count'Image & 
                  " does not match tensor shape size" & Expected_Size'Image);
      end if;
   end Validate_Dimensions;
   
  function Get_YAML_Array
  (YAML_Content : String;
   Field        : String) return String 
is
   Field_Start : constant String := Field & ":";
   Start_Pos   : Natural := Index(YAML_Content, Field_Start);
   End_Pos     : Natural;
   Next_Field  : Natural;
begin
   if Start_Pos = 0 then
      raise YAML_Parse_Error with "Field '" & Field & "' not found in YAML";
   end if;
   
   Start_Pos := Start_Pos + Field_Start'Length;
   
   Next_Field := Index(YAML_Content(Start_Pos..YAML_Content'Last), ASCII.LF & "labels:");
   if Next_Field = 0 then
      End_Pos := YAML_Content'Last;
   else
      End_Pos := Start_Pos + Next_Field - 2;
   end if;
   
   while End_Pos > Start_Pos and then
         (YAML_Content(End_Pos) = ' ' or 
          YAML_Content(End_Pos) = ASCII.LF or 
          YAML_Content(End_Pos) = ASCII.CR) loop
      End_Pos := End_Pos - 1;
   end loop;
   
   --Put_Line("Debug: Extracted '" & Field & "' section = '" & YAML_Content(Start_Pos..End_Pos) & "'");
   
   return YAML_Content(Start_Pos..End_Pos);
end Get_YAML_Array;
   
   function From_YAML_Array
     (YAML_Data : String;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
      Result    : Tensor_T := Zeros(Shape);
      Elements  : Elements_T := Parse_Coordinate_Data(YAML_Data);
      Index     : Natural := 1;
   begin
      -- Convert flat YAML array to tensor based on shape
      for I in 1 .. Shape(1) loop
         for J in 1 .. Shape(2) loop
            if Index <= Elements'Last then
               Result.Set((I, J), Elements(Index));
               Index := Index + 1;
            end if;
         end loop;
      end loop;
      
      return Result;
   end From_YAML_Array;
   
   function Load_YAML_Tensor
     (Filename : String;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
      File    : File_Type;
      Content : Unbounded_String;
   begin
      -- Read YAML file
      Open(File, In_File, Filename);
      while not End_Of_File(File) loop
         Append(Content, Get_Line(File) & ASCII.LF);
      end loop;
      Close(File);
      
      -- Parse YAML and convert to tensor
      return Parse_YAML_Tensor(To_String(Content), Shape);
   exception
      when E : others =>
         if Is_Open(File) then
            Close(File);
         end if;
         raise YAML_Parse_Error with
           "Error loading YAML file: " & Exception_Message(E);
   end Load_YAML_Tensor;

   function Parse_YAML_Array_Items
  (YAML_Array : String) return Elements_T 
is
begin
   -- For YAML data with coordinate pairs, use the specialized function
   if Index(YAML_Array, "[") > 0 then
      return Parse_Coordinate_Data(YAML_Array);
   else
      -- For YAML data with simple values (like labels), use the label parser
      return Parse_Label_Data(YAML_Array);
   end if;
end Parse_YAML_Array_Items;
   
   function Parse_YAML_Tensor
     (YAML_Str : String;
      Shape    : Tensor_Shape_T) return Tensor_T
   is
   begin
      return From_YAML_Array(YAML_Str, Shape);
   exception
      when E : others =>
         raise YAML_Parse_Error with
           "Error parsing YAML string: " & Exception_Message(E);
   end Parse_YAML_Tensor;
   
function Load_Dataset
     (Filename     : String;
      Data_Shape   : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T) return Del.JSON.Dataset_Array
   is
      File    : File_Type;
      Content : Unbounded_String;
   begin
      Open(File, In_File, Filename);
      while not End_Of_File(File) loop
         Append(Content, Get_Line(File) & ASCII.LF);
      end loop;
      Close(File);
      
      declare
         YAML_Content : constant String := To_String(Content);
         Data_Section : constant String := Get_YAML_Array(YAML_Content, "data");
         Label_Section : constant String := Get_YAML_Array(YAML_Content, "labels");
         
         Data_Items : Elements_T := Parse_Coordinate_Data(Data_Section);
         Label_Items : Elements_T := Parse_Label_Data(Label_Section);
      begin
         if Label_Items'Length = 0 then
            raise YAML_Parse_Error with "No labels parsed from YAML file";
         end if;
         
         declare
            Num_Samples : constant Positive := Label_Items'Length;
            Dataset : Dataset_Array(1 .. Num_Samples);
            Features_Per_Sample : constant Positive := Data_Shape(2);
         begin
            Put_Line("Parsed" & Data_Items'Length'Image & " data elements and" & 
                     Label_Items'Length'Image & " labels");
            
            if Data_Items'Length /= Num_Samples * Features_Per_Sample then
               Put_Line("Warning: Data items length: " & Data_Items'Length'Image);
            end if;
            
            for I in 1 .. Num_Samples loop
               declare
                  Sample_Data : Tensor_T := Zeros(Data_Shape);
                  Index : Natural := (I - 1) * Features_Per_Sample + 1;
               begin
                  for J in 1 .. Features_Per_Sample loop
                     if Index <= Data_Items'Length then
                        Sample_Data.Set((1, J), Data_Items(Index));
                        Index := Index + 1;
                     end if;
                  end loop;
                  Dataset(I).Data := new Tensor_T'(Sample_Data);
               end;
               
               declare
                  Label : constant Integer := Integer(Label_Items(I));
                  Target : Tensor_T := Zeros(Target_Shape);
               begin
                  if Label < 1 or Label > Target_Shape(2) then
                     raise YAML_Parse_Error with 
                        "Label " & Label'Image & " out of bounds for Target_Shape " & Target_Shape(2)'Image;
                  end if;
                  Target.Set((1, Label), 1.0);
                  Dataset(I).Target := new Tensor_T'(Target);
               end;
            end loop;
            
            return Dataset;
         end;
      end;
   exception
      when YAML_Parse_Error =>
         if Is_Open(File) then
            Close(File);
         end if;
         raise;
      when E : others =>
         if Is_Open(File) then
            Close(File);
         end if;
         Put_Line("Error loading dataset: " & Exception_Message(E));
         raise YAML_Parse_Error with "Error loading dataset: " & Exception_Message(E);
   end Load_Dataset;
   
end Del.YAML;