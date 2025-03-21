with Ada.Strings.Unbounded;
with Ada.Containers;
with Del.JSON; use Del.JSON;  -- Use the Dataset types from JSON package

package Del.YAML is
   -- Reuse Dataset_Entry and Dataset_Array types from Del.JSON
   
   -- YAML parsing error
   YAML_Parse_Error : exception;
   
   -- Convert YAML data to tensor
   function From_YAML_Array
     (YAML_Data : String;
      Shape     : Tensor_Shape_T) return Tensor_T;
   
   -- Load YAML file and convert to tensor
   function Load_YAML_Tensor
     (Filename : String;
      Shape    : Tensor_Shape_T) return Tensor_T;
   
   -- Parse YAML string to tensor
   function Parse_YAML_Tensor
     (YAML_Str : String;
      Shape    : Tensor_Shape_T) return Tensor_T;
      
   -- Load dataset from YAML file
   function Load_Dataset
     (Filename     : String;
      Data_Shape   : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T) return Del.JSON.Dataset_Array;
      
   -- Parse array of the form [x, y] to Elements_T
   function To_Element_Array(Value : String) return Elements_T;
   
private
   -- Helper function to validate YAML data dimensions
   procedure Validate_Dimensions
     (YAML_Data : String;
      Shape     : Tensor_Shape_T);
   
   -- Helper to convert YAML value to Element_T
   function To_Element(Value : String) return Element_T;
   
   -- Helper to extract array from YAML object
   function Get_YAML_Array
     (YAML_Content : String;
      Field        : String) return String;
      
   -- Helper to parse YAML array items
   function Parse_YAML_Array_Items
     (YAML_Array : String) return Elements_T;
end Del.YAML;