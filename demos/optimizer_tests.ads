with AUnit.Test_Cases;

package Optimizer_Tests is
   type Test_Case is new AUnit.Test_Cases.Test_Case with null record;

   -- Overriding the Name function
   overriding function Name (T : Test_Case) return String;

end Optimizer_Tests;
