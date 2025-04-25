with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics; use Ada.Numerics;
with Del; use Del;
with Del.Loss;
with Del.Operators;
with Del.Model;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Orka; use Orka;

procedure Loss_Testcases is
   package DL renames Del.Loss;
   package DOp renames Del.Operators;
   package DMod renames Del.Model;

   -- Test Tensors
   Expected : constant Tensor_T := To_Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]); -- One-hot expected labels
   Actual   : constant Tensor_T := To_Tensor([0.7, 0.2, 0.1, 0.1, 0.8, 0.1], [2, 3]); -- Predicted logits
   Perfect_Predictions : constant Tensor_T := To_Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]); -- Perfect predictions

   -- Cross-Entropy Loss instance
   CE : DL.Cross_Entropy_T;

   -- Network and Layers
   Network : DMod.Model;
   Linear_Layer : DOp.Linear_Access_T := new DOp.Linear_T;

   -- Expected Values for Validation
   Expected_Loss : constant Float := 0.3566749; -- Precomputed expected loss
   Tolerance : constant Float := 0.0001;

   -- Helper Procedure for Assertions (Float-based)
   procedure Assert_Float(Expected, Actual : Float; Test_Name : String) is
   begin
      if Abs(Expected - Actual) > Tolerance then
         Put_Line(Test_Name & " Failed");
         Put_Line("Expected: " & Float'Image(Expected));
         Put_Line("Actual  : " & Float'Image(Actual));
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Float;

   -- Helper Procedure for Tensor Shape Assertion
   procedure Assert_Shape(Expected, Actual : Tensor_T; Test_Name : String) is
   begin
      if Shape(Expected) /= Shape(Actual) then
         Put_Line(Test_Name & " Failed - Shape mismatch");
         Put_Line("Expected Shape: " & Shape(Expected)'Image);
         Put_Line("Actual Shape  : " & Shape(Actual)'Image);
      else
         Put_Line(Test_Name & " Passed");
      end if;
   end Assert_Shape;

begin
   Put_Line("=== Cross-Entropy Loss Testcases ===");

   -- 1. Test Forward Loss computation
   Put_Line("1. Testing Forward Loss (Basic)");
   declare
      Loss_Result : constant Float := CE.Forward(Expected, Actual);
   begin
      Put_Line("Computed Loss: " & Float'Image(Loss_Result));
      Assert_Float(Expected_Loss, Loss_Result, "Forward Loss Test");
   end;

   -- 2. Test Forward Loss for Perfect Predictions
   Put_Line("2. Testing Forward Loss (Perfect Predictions)");
   declare
      Perfect_Loss : constant Float := CE.Forward(Expected, Perfect_Predictions);
   begin
      Put_Line("Computed Loss (Perfect Predictions): " & Float'Image(Perfect_Loss));
      if Perfect_Loss < 1.0E-5 then
         Put_Line("Perfect Prediction Loss Test Passed");
      else
         Put_Line("Perfect Prediction Loss Test Failed");
      end if;
   end;

   -- 3. Test Backward Gradient
   Put_Line("3. Testing Backward Gradient Calculation");
   declare
      Grad : constant Tensor_T := CE.Backward(Expected, Actual);
   begin
      Put_Line("Gradient Tensor:");
      Put_Line(Grad.Image);
      Assert_Shape(Expected, Grad, "Gradient Shape Test");
   end;

   -- 4. Test Loss inside a Model
   Put_Line("4. Testing Loss Computation inside a Model");
   declare
      Network_Input : Tensor_T := To_Tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [2, 3]);
      Network_Output : Tensor_T := Zeros([2, 3]);
      Computed_Loss : Float;
   begin
      -- Create simple network
      Linear_Layer.Initialize(3, 3);
      DMod.Add_Layer(Network, Func_Access_T(Linear_Layer));

      -- Run Network
      Network_Output := Network.Run_Layers(Network_Input);
      Put_Line("Network Output:");
      Put_Line(Network_Output.Image);

      -- Compute Loss
      Computed_Loss := CE.Forward(Expected, Network_Output);
      Put_Line("Loss After Network Forward: " & Float'Image(Computed_Loss));
   end;

   Put_Line("=== All Loss Testcases Completed ===");
end Loss_Testcases;
