with AUnit.Test_Suites;
with AUnit.Run;
with Model_Suite;

use AUnit.Test_Suites;
use AUnit.Run;

procedure Mlengine_Tests is
   Main_Suite : constant Test_Suite_Access := New_Root_Suite ("Main Test Suite");
   Options    : constant Options := Default_Options;
begin
   -- Register model tests
   Main_Suite.Register_Suite (Model_Suite.Get_Test_Suites);

   -- Run all registered tests
   Run (Main_Suite, Options);
end Mlengine_Tests;
