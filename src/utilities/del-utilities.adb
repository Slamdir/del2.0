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

end Del.Utilities;
