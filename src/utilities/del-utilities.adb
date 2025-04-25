with Ada.Numerics.Discrete_Random;

package body Del.Utilities is

   function Generate_Random_List(x : Positive) return Integer_Array is      
      subtype Rand_Range is Positive range 1..x;
      package Random_Int is new Ada.Numerics.Discrete_Random(Rand_Range);
      Gen : Random_Int.Generator;
      
      List : Integer_Array(1..x);
      Temp : Integer;
      K    : Rand_Range;
   begin
      -- Initialize list with values 1 to x
      for I in List'Range loop
         List(I) := I;
      end loop;
      
      Random_Int.Reset(Gen);  -- Seed the generator
      
      -- Perform Fisher-Yates shuffle
      for I in reverse List'Range loop
         K := Random_Int.Random(Gen);  -- Generates random index in 1..I
         Temp := List(I);
         List(I) := List(K);
         List(K) := Temp;
      end loop;
      
      return List;
   end Generate_Random_List;

   procedure Print_Array(arr : Integer_Array) is
   begin
      Put_Line("*******************************************");
      Put_Line("*            Indecies                     *");
      Put_Line("*******************************************");
      Put_Line("\n\n");
      Put("[");
      for I in arr'Range loop
         Put(Integer'Image(arr(I)));
         if I /= arr'Last then
            Put(", ");
         end if;
      end loop;
      Put_Line("]");
   end Print_Array;


   function Compute_Num_Accurate (
      Actual_Labels    : Tensor_T;
      Predicted_Labels : Tensor_T
   ) return Natural is
      Actual_Array : Integer_Array := Labels_To_Int_Array (Actual_Labels);
      Predicted_Array : Integer_Array := Labels_To_Int_Array (Predicted_Labels);
      Labels : Natural := Actual_Array'Length;
      Correct_Predictions : Natural := 0;
   begin
      for I in 1..Labels loop
         if Actual_Array(I) = Predicted_Array(I) then
            Correct_Predictions := Correct_Predictions + 1;
         end if;
      end loop;
      return Correct_Predictions;      
   end Compute_Num_Accurate;

   function Labels_To_Int_Array (
      Labels : Tensor_T
   ) return Integer_Array is
      Rows        : constant Integer := Shape(Labels)(1);
      Columns     : constant Integer := Shape(Labels)(2);
      Results     : Integer_Array(1..Rows);
   begin
      for I in 1..Rows loop
         declare
            max_Column : Natural := 1;
            max_Value : Element_T := 0.0;
         begin
            for J in 1..Columns loop
               declare
                  Label  : Element_T := Labels([I, J]);
               begin
                  if Label > max_Value then
                     max_Value := Label;
                     max_Column := J;
                  end if;
               end;
            end loop;
            Results(I) := max_Column;
         end;
      end loop;
      return Results;      
   end Labels_To_Int_Array;

end Del.Utilities;
