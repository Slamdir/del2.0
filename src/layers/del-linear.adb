with Ada.Numerics.Float_Random;

package body Del.Linear is

   -- Renaming Ada.Numerics.Float_Random for convenience
   package Random_Float renames Ada.Numerics.Float_Random;

   procedure Initialize_Layer (Layer : out Linear_Layer) is
      -- Random number generator
      Gen : Random_Float.Generator;
   begin
      -- Initialize the random number generator
      Random_Float.Reset (Gen);

      -- Initialize Weights with random values
      for I in Layer.Weights'Range (1) loop
         for J in Layer.Weights'Range (2) loop
            Layer.Weights (I, J) := Random_Float.Random (Gen);
         end loop;
      end loop;

      -- Initialize Bias with zeros
      for I in Layer.Bias'Range loop
         Layer.Bias (I) := 0.0;
      end loop;
   end Initialize_Layer;

   function Forward (Input : Float_Array; Layer : Linear_Layer) return Float_Array is
      Output : Float_Array (Layer.Bias'Range) := (others => 0.0);
   begin
      -- Simple linear transformation: Output = Weights * Input + Bias
      for J in Output'Range loop
         for I in Input'Range loop
            Output (J) := Output (J) + Layer.Weights (I, J) * Input (I);
         end loop;
         Output (J) := Output (J) + Layer.Bias (J);
      end loop;
      return Output;
   end Forward;

   procedure Backward (Input : Float_Array; d_Output : Float_Array; Layer : in out Linear_Layer) is
      Learning_Rate : constant Float := 0.01;
   begin
      -- Update Weights and Bias using gradient descent
      for I in Input'Range loop
         for J in d_Output'Range loop
            Layer.Weights (I, J) := Layer.Weights (I, J)
              - Learning_Rate * d_Output (J) * Input (I);
         end loop;
      end loop;

      for J in d_Output'Range loop
         Layer.Bias (J) := Layer.Bias (J) - Learning_Rate * d_Output (J);
      end loop;
   end Backward;

end Del.Linear;
