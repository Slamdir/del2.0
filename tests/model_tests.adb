with AUnit.Assertions; use AUnit.Assertions;
with Model;

package body Model_Tests is
   -- Test forward pass
   procedure Test_Forward (T : in out AUnit.Test_Cases.Test_Case'Class) is
      My_Model : Model.Model_T;
      Input : Tensor;
      Expected_Output : Tensor;
      Actual_Output : Tensor;
   begin
      -- Initialize the model
      Model.Initialize(My_Model, Layers => 2, Units => (2, 2));

      -- Define input and expected output
      Input := ST_CPU.To_Tensor((1.0, 2.0), (1, 2));
      Expected_Output := ST_CPU.To_Tensor((10));  -- Replace with expected result

      -- Perform forward pass and assert equality
      Actual_Output := Model.Forward(My_Model, Input);
      Assert_Equal(Actual_Output, Expected_Output, "Forward pass failed.");
   end Test_Forward;

   -- Test backward pass
   procedure Test_Backward (T : in out AUnit.Test_Cases.Test_Case'Class) is
      My_Model : Model.Model_T;
      Input : Tensor;
      Gradient : Tensor;
   begin
      -- Initialize model
      Model.Initialize(My_Model, Layers => 2, Units => (2, 2));

      -- Perform forward and backward pass
      Input := ST_CPU.To_Tensor((1.0, 2.0), (1, 2));
      Gradient := ST_CPU.To_Tensor((0.5, -0.5), (1, 2));
      Model.Forward(My_Model, Input);
      Model.Backward(My_Model, Gradient);

      -- Assertions for gradient values (optional)
      Assert_True(..., "Backward pass failed.");
   end Test_Backward;

   -- Test end-to-end training
   procedure Test_End_To_End (T : in out AUnit.Test_Cases.Test_Case'Class) is
      My_Model : Model.Model_T;
      Input : Tensor;
      Target : Tensor;
      Loss_Before : Float;
      Loss_After : Float;
   begin
      -- Initialize the model
      Model.Initialize(My_Model, Layers => 2, Units => (2, 2));

      -- Define input and target
      Input := ST_CPU.To_Tensor((1.0, 2.0), (1, 2));
      Target := ST_CPU.To_Tensor((0.0, 1.0), (1, 2));

      -- Compute loss before training
      Model.Forward(My_Model, Input);
      Loss_Before := Model.Compute_Loss(My_Model, Target);

      -- Perform backward pass and update parameters
      Model.Backward(My_Model, Model.Compute_Gradients(Target));
      Model.Update_Parameters(My_Model);

      -- Compute loss after training
      Model.Forward(My_Model, Input);
      Loss_After := Model.Compute_Loss(My_Model, Target);

      -- Assert loss decreases
      Assert(Loss_After < Loss_Before, "End-to-end training failed: Loss did not decrease.");
   end Test_End_To_End;

end Model_Tests;
