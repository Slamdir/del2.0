with AUnit.Test_Cases;

package Model_Tests is
   -- Test case for the forward pass
   procedure Test_Forward (T : in out AUnit.Test_Cases.Test_Case'Class);

   -- Test case for the backward pass
   procedure Test_Backward (T : in out AUnit.Test_Cases.Test_Case'Class);

   -- Test case for end-to-end model training
   procedure Test_End_To_End (T : in out AUnit.Test_Cases.Test_Case'Class);
end Model_Tests;
