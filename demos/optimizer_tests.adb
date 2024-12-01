with AUnit.Assertions; use AUnit.Assertions;
with Del.Optimizer;
with Orka.Numerics.Singles.Tensors.CPU;

package body Optimizer_Tests is

   procedure Test_SGD_Optimizer (R : in out AUnit.Test_Cases.Test_Case'Class);

   -- Name function implementation
   overriding function Name (T : Test_Case) return String is
   begin
      return "Optimizer Test Case";
   end Name;

   -- Register the test routine
   procedure Register_Tests (T : in out Test_Case) is
      use AUnit.Test_Cases.Registration;
   begin
      Register_Routine (T, Test_SGD_Optimizer'Access, "Test SGD Optimizer");
   end Register_Tests;

   -- Test routine for SGD Optimizer
   procedure Test_SGD_Optimizer (R : in out AUnit.Test_Cases.Test_Case'Class) is
      package DOpt renames Del.Optimizer;
      package TCPU renames Orka.Numerics.Singles.Tensors.CPU;

      Optimizer  : DOpt.SGD_Optimizer_T;
      Params     : TCPU.Tensor_T := TCPU.To_Tensor([5.0, 10.0, 15.0], [3]);
      Gradients  : TCPU.Tensor_T := TCPU.To_Tensor([0.5, 1.0, 1.5], [3]);
      Expected   : TCPU.Tensor_T := TCPU.To_Tensor([4.95, 9.9, 14.85], [3]);
   begin
      -- Set learning rate
      DOpt.Set_Learning_Rate(Optimizer, 0.1);

      -- Apply optimizer step
      DOpt.Step(Optimizer, Params, Gradients);

      -- Assert that the updated parameters match the expected values
      for I in Params'Range(1) loop
         Assert
           (Params.Get(I) = Expected.Get(I),
            "Parameter " & I'Image & " mismatch. Got: " & Params.Get(I)'Image &
            " Expected: " & Expected.Get(I)'Image);
      end loop;
   end Test_SGD_Optimizer;

end Optimizer_Tests;
