with Ada.Text_IO; use Ada.Text_IO;
with Del;
with Del.Operators;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Integration_Suite is
   package D renames Del;
   package DOp renames Del.Operators;

   -- Test Inputs
   Input : D.Tensor_T := To_Tensor([1.0, 2.0], [1, 2]);  -- Single input vector
   Batch_Input : D.Tensor_T := To_Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]);  -- Batch input

   -- Layers
   Linear_Layer : DOp.Linear_T;
   ReLU_Layer : DOp.ReLU_T;
   Softmax_Layer : DOp.SoftMax_T;

   -- Helper procedure to assert equality
   procedure Assert_Equal(Expected, Actual : D.Tensor_T; Test_Name : String) is
      Expected_Image : constant String := Expected.Image;
      Actual_Image   : constant String := Actual.Image;
   begin
      if Expected_Image /= Actual_Image then
         Put_Line("Test Failed: " & Test_Name);
         Put_Line("Expected: " & Expected_Image);
         Put_Line("Actual  : " & Actual_Image);
      else
         Put_Line("Test Passed: " & Test_Name);
      end if;
   end Assert_Equal;

begin
   Put_Line("=== Integration Tests ===");

   -- Linear + ReLU
   Put_Line("1. Testing Linear + ReLU");
   Linear_Layer.Initialize(2, 2);
   declare
      Linear_Output : D.Tensor_T := Linear_Layer.Forward(Input);
      Final_Output : D.Tensor_T := ReLU_Layer.Forward(Linear_Output);
   begin
      Assert_Equal(
         Expected => To_Tensor([0.0, 1.5], [1, 2]),  -- Example ReLU output
         Actual   => Final_Output,
         Test_Name => "Linear + ReLU"
      );
   end;

   -- Linear + Softmax
   Put_Line("2. Testing Linear + Softmax");
   Linear_Layer.Initialize(2, 3);  -- Change output size for logits
   declare
      Linear_Output : D.Tensor_T := Linear_Layer.Forward(Input);
      Final_Output : D.Tensor_T := Softmax_Layer.Forward(Linear_Output);
   begin
      Assert_Equal(
         Expected => To_Tensor([0.2, 0.3, 0.5], [1, 3]),  -- Example softmax output
         Actual   => Final_Output,
         Test_Name => "Linear + Softmax"
      );
   end;

   -- Linear + ReLU + Softmax
   Put_Line("3. Testing Linear + ReLU + Softmax");
   Linear_Layer.Initialize(2, 3);  -- Output size for logits
   declare
      Linear_Output : D.Tensor_T := Linear_Layer.Forward(Input);
      ReLU_Output   : D.Tensor_T := ReLU_Layer.Forward(Linear_Output);
      Final_Output  : D.Tensor_T := Softmax_Layer.Forward(ReLU_Output);
   begin
      Assert_Equal(
         Expected => To_Tensor([0.1, 0.4, 0.5], [1, 3]),  -- Example softmax output
         Actual   => Final_Output,
         Test_Name => "Linear + ReLU + Softmax"
      );
   end;

   Put_Line("=== All Integration Tests Completed ===");
end Integration_Suite;
