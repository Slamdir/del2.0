with Del.JSON; use Del.JSON;

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

   -- Future expansion could include other data formats:
   -- function Load_From_CSV
   -- etc.

private
   type Training_Data is tagged record
      Data   : Tensor_Access_T;
      Labels : Tensor_Access_T;
   end record;

end Del.Data;