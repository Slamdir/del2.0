with AUnit.Run; use AUnit.Run;
with Optimizer_Tests;

procedure Main_Test is
begin
   -- Register the test case
   Register_Test_Case(new Optimizer_Tests.Test_Case);

   -- Run all registered tests
   Run_All_Tests;
end Main_Test;
