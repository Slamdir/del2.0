with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics; use Ada.Numerics;
with Del; use Del;
with Del.Loss;
with Del.Operators;
with Del.Model;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Loss_Test is
   package DL renames Del.Loss;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test Tensors
   Expected : Tensor_T := To_Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]); -- One-hot encoded expected values
   Actual   : Tensor_T := To_Tensor([0.7, 0.2, 0.1, 0.1, 0.8, 0.1], [2, 3]); -- Predicted logits

   -- Cross-Entropy Loss Instance
   CE : DL.Cross_Entropy_T;

   -- Network and Layers
   Network : DMod.Model;
   Linear_Layer : DOp.Linear_Access_T := new DOp.Linear_T;

   -- Expected Values for Validation
   Expected_Loss : constant Float := 0.3566749; -- Precomputed expected loss
   Tolerance : constant Float := 0.0001;

   -- Helper Procedure for Assertions
   procedure Assert_Test(Expected, Actual : Float; Test_Name : String) is
   begin
      if Abs(Expected - Actual) > Tolerance then
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected: " & Float'Image(Expected));
         Put_Line("Actual  : " & Float'Image(Actual));
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Test;

begin
   Put_Line("=== Cross-Entropy Loss Tests ===");

   -- Test Forward Method
   Put_Line("1. Testing Forward Method");
   declare
      Loss_Result : constant Float := Float(CE.Forward(Expected, Actual));
   begin
      Put_Line("Computed Loss: " & Float'Image(Loss_Result));
      Assert_Test(Expected_Loss, Loss_Result, "Forward Method Test");
   end;

   -- Add to Network and Test
   Put_Line("2. Adding Loss to Network");
   begin
      -- Initialize and add a linear layer to the network
      Linear_Layer.Initialize(3, 3);
      DMod.Add_Layer(Network, Del.Func_Access_T(Linear_Layer));

      -- Run forward pass
      declare
         Network_Input : Tensor_T := To_Tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [2, 3]);
         Network_Output : Tensor_T := Network.Run_Layers(Network_Input);
         Computed_Loss : constant Float := Float(CE.Forward(Expected, Network_Output));
      begin
         Put_Line("Network Output:");
         Put_Line(Network_Output.Image);
         Put_Line("Loss After Network Forward Pass: " & Float'Image(Computed_Loss));
      end;
   exception
      when others =>
         Put_Line("Error during Loss Integration with Network");
   end;

   Put_Line("=== Cross-Entropy Loss Tests Completed ===");
end Loss_Test;
