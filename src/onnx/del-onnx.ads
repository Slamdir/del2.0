with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Del.Model;

package Del.ONNX is
   ONNX_Error : exception;
   
   -- ONNX types mapping
   type ONNX_Data_Type is (
      UNDEFINED,
      FLOAT,
      UINT8,
      INT8,
      UINT16,
      INT16,
      INT32,
      INT64,
      STRING_TYPE,  -- Renamed from STRING to avoid conflict
      BOOL
   );

   -- ONNX node types we support
   type ONNX_Op_Type is (
      Linear,
      ReLU,
      Unknown
   );

   
   Max_IO_Count : constant := 10;
   subtype IO_Index is Positive range 1 .. Max_IO_Count;
   type Node_String_Array is array (IO_Index) of Unbounded_String;

   -- Represents an ONNX node
   type ONNX_Node is record
      Op_Type : ONNX_Op_Type;
      Name : Unbounded_String;
      Inputs : Node_String_Array;
      Outputs : Node_String_Array;
      Input_Count : Natural := 0;
      Output_Count : Natural := 0;
   end record;

   -- Core procedures
   procedure Load_ONNX_Model(
      Model : in out Del.Model.Model;
      Filename : String);

private
   -- Internal procedures
   procedure Parse_ONNX_Binary(
      Filename : String;
      Model : in out Del.Model.Model);
      
   function Create_Layer_From_Node(
      Node : ONNX_Node) return Func_Access_T;
      
   procedure Add_Node_To_Model(
      Model : in out Del.Model.Model;
      Node : ONNX_Node);
end Del.ONNX;