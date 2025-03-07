with GNATCOLL.JSON; use GNATCOLL.JSON;
with Ada.Strings.Unbounded;
with Ada.Containers;

package Del.JSON is
   type Dataset_Entry is record
      Data   : Tensor_Access_T;
      Target : Tensor_Access_T;
   end record;
   
   type Dataset_Array is array (Positive range <>) of Dataset_Entry;
   
   -- Convert JSON array to tensor
   function From_JSON_Array
     (JSON_Data : JSON_Array;
      Shape    : Tensor_Shape_T) return Tensor_T;
   
   -- Load JSON file and convert to tensor
   function Load_JSON_Tensor
     (Filename : String;
      Shape    : Tensor_Shape_T) return Tensor_T;
   
   -- Parse JSON string to tensor
   function Parse_JSON_Tensor
     (JSON_Str : String;
      Shape    : Tensor_Shape_T) return Tensor_T;
      
   -- Load dataset from JSON file
   function Load_Dataset
     (Filename     : String;
      Data_Shape   : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T) return Dataset_Array;
   
   -- JSON parsing error
   JSON_Parse_Error : exception;
   
private
   -- Helper function to validate JSON array dimensions
   procedure Validate_Dimensions
     (JSON_Data : JSON_Array;
      Shape     : Tensor_Shape_T);
   
   -- Helper to convert JSON value to Element_T
   function To_Element(Value : JSON_Value) return Element_T;
   
   -- Helper to extract array from JSON object
   function Get_JSON_Array
     (Object : JSON_Value;
      Field  : String) return JSON_Array;
end Del.JSON;