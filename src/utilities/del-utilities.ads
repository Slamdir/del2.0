package Del.Utilities is

   type Integer_Array is array (Positive range <>) of Integer;
   
   function Generate_Random_List(x : Positive) return Integer_Array;

   procedure Print_Array(arr : Integer_Array);

end Del.Utilities;
