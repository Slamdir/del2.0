with AUnit.Test_Cases.Registration;
with Model_Tests;

package body Model_Suite is
   procedure Register (Suite : in out AUnit.Test_Cases.Test_Suite'Class) is
   begin
      -- Register each model-related test case here
      Register_Routine(Suite, Model_Tests.Test_Forward'Access, "Test Forward Pass");
      Register_Routine(Suite, Model_Tests.Test_Backward'Access, "Test Backward Pass");
      Register_Routine(Suite, Model_Tests.Test_End_To_End'Access, "Test End-to-End Training");
   end Register;
end Model_Suite;
