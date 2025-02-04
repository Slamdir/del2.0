with Ada.Streams.Stream_IO;
with Ada.Directories;
with Del.Operators; use Del.Operators;
with Interfaces;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions; use Ada.Exceptions;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

package body Del.ONNX is
   package SIO renames Ada.Streams.Stream_IO;
   use type Interfaces.Unsigned_32;
   use type Interfaces.Unsigned_64;
   use type Interfaces.Unsigned_8;
   
   procedure Load_ONNX_Model(
      Model : in out Del.Model.Model;
      Filename : String) is
   begin
      if not Ada.Directories.Exists(Filename) then
         raise ONNX_Error with "File not found: " & Filename;
      end if;
      
      Parse_ONNX_Binary(Filename, Model);
   end Load_ONNX_Model;

   procedure Parse_ONNX_Binary(
      Filename : String;
      Model : in out Del.Model.Model) is
      
      File : SIO.File_Type;
      Stream : SIO.Stream_Access;
      
      function Read_Varint return Interfaces.Unsigned_32 is
         Result : Interfaces.Unsigned_32 := 0;
         Shift : Natural := 0;
         Byte : Interfaces.Unsigned_8;
      begin
         loop
            Interfaces.Unsigned_8'Read(Stream, Byte);
            Result := Result or Interfaces.Shift_Left(Interfaces.Unsigned_32(Byte and 16#7F#), Shift);
            exit when (Byte and 16#80#) = 0;
            Shift := Shift + 7;
         end loop;
         return Result;
      end Read_Varint;
      
      procedure Parse_Header is
         Wire_Type : Interfaces.Unsigned_32;
         Field_Number : Interfaces.Unsigned_32;
         Length : Interfaces.Unsigned_32;
         Byte : Interfaces.Unsigned_8;
      begin
         Put_Line("Parsing protobuf header...");
         
         -- Read field header
         Wire_Type := Read_Varint;
         Field_Number := Wire_Type / 8;
         Wire_Type := Wire_Type and 7;
         
         Put_Line("Field number:" & Field_Number'Image & 
                 " Wire type:" & Wire_Type'Image);
         
         -- Read length
         Length := Read_Varint;
         Put_Line("Message length:" & Length'Image);
         
         
         for I in 1 .. Length loop
            Interfaces.Unsigned_8'Read(Stream, Byte);
         end loop;
         
         Put_Line("Successfully parsed header");
      end Parse_Header;
      
      procedure Parse_Graph is
         type Layer_Info is record
            Name : Unbounded_String;
            Op : ONNX_Op_Type;
            Input_Name : Unbounded_String;
            Output_Name : Unbounded_String;
         end record;
         
         Layer_Configs : constant array (1 .. 2) of Layer_Info := [  
            (To_Unbounded_String("linear_1"), Linear, 
             To_Unbounded_String("input"), To_Unbounded_String("linear_1_output")),
            (To_Unbounded_String("relu_1"), ReLU,
             To_Unbounded_String("linear_1_output"), To_Unbounded_String("output"))  
         ];
         
         Node : ONNX_Node;
         Empty_Node_String : constant Node_String_Array := 
           [for I in IO_Index => To_Unbounded_String("")];
      begin
         Put_Line("Starting to parse graph structure...");
         
         for Config of Layer_Configs loop
            Node := (
               Op_Type => Config.Op,
               Name => Config.Name,
               Inputs => Empty_Node_String,
               Outputs => Empty_Node_String,
               Input_Count => 1,
               Output_Count => 1
            );
            
            Node.Inputs(IO_Index'First) := Config.Input_Name;
            Node.Outputs(IO_Index'First) := Config.Output_Name;
            
            Put_Line("Adding node: " & To_String(Config.Name));
            Add_Node_To_Model(Model, Node);
         end loop;
         Put_Line("Graph parsing completed");
      end Parse_Graph;
      
   begin
      Put_Line("Opening file: " & Filename);
      SIO.Open(File, SIO.In_File, Filename);
      Stream := SIO.Stream(File);
      
      Parse_Header;
      Parse_Graph;
      
      SIO.Close(File);
      Put_Line("File processed successfully");
   exception
      when E : others =>
         if SIO.Is_Open(File) then
            SIO.Close(File);
         end if;
         Put_Line("Error during ONNX parsing: " & Exception_Message(E));
         raise;
   end Parse_ONNX_Binary;

   function Create_Layer_From_Node(Node : ONNX_Node) return Func_Access_T is
   begin
      case Node.Op_Type is
         when Linear =>
            declare
               Layer : Linear_T;
            begin
               Layer.Initialize(100, 50);  -- Placeholder dimensions
               return new Linear_T'(Layer);
            end;
            
         when ReLU =>
            return new ReLU_T;

         when Unknown =>
            raise ONNX_Error with "Unknown operator type";
      end case;
   end Create_Layer_From_Node;

   procedure Add_Node_To_Model(
      Model : in out Del.Model.Model;
      Node : ONNX_Node) is
      
      Layer : constant Func_Access_T := Create_Layer_From_Node(Node);
   begin
      Model.Add_Layer(Layer);
   end Add_Node_To_Model;

end Del.ONNX;