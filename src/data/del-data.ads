with Del.JSON; use Del.JSON;
with Del.YAML; use Del.YAML;

package Del.Data is
   -- Training data type definition
   type Training_Data is tagged private;
   type Training_Data_Access is access all Training_Data'Class;

   -- Constructor for Training_Data
   function Create
     (Data   : Tensor_T;
      Labels : Tensor_T) return Training_Data_Access;

   -- Accessors for Training_Data
   function Get_Data(Self : Training_Data) return Tensor_T;
   function Get_Labels(Self : Training_Data) return Tensor_T;

   -- Data loading functions
   function Load_From_JSON
     (JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access;
      
   -- New YAML loading function
   function Load_From_YAML
     (YAML_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access;
      
   -- Generic load function that detects file type by extension
   function Load_From_File
     (Filename      : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T) return Training_Data_Access;

   function Get_All_Samples
      (Self : Training_Data) return Dataset_Array;


   -- Future expansion could include other data formats:
   -- function Load_From_CSV
   -- etc.

private
   type Dataset_Array_Access is access all Dataset_Array;
   type Training_Data is tagged record
      Data   : Tensor_Access_T;
      Labels : Tensor_Access_T;
      Dataset : Dataset_Array_Access;
   end record;

end Del.Data;