package Del.Utilities is

   -- Shuffle
   type Integer_Array is array (Positive range <>) of Integer;
   
   function Generate_Random_List(x : Positive) return Integer_Array;
   procedure Print_Array(arr : Integer_Array);

   -- Metrics
   function Compute_Num_Accurate (
      Actual_Labels    : Tensor_T;
      Predicted_Labels : Tensor_T
   ) return Natural;

   function Labels_To_Int_Array (Labels : Tensor_T) return Integer_Array;

end Del.Utilities;
