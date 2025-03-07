with Del;
with Del.Operators;
with Del.Model;
with Del.Initializers;
with Del.Loss;
with Del.Optimizers;
with Export_Model;
with Ada.Text_IO; use Ada.Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles; use Orka.Numerics.Singles;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure Main_Demo is

   -- Define input and labels for supervised learning
   Input  : Del.Tensor_T := To_Tensor([
      Float_32(9.0), Float_32(2.0), Float_32(-4.0),
      Float_32(-5.0), Float_32(5.0), Float_32(0.0),
      Float_32(3.0), Float_32(15.0), Float_32(9.0)], [3,3]);

   Labels : Del.Tensor_T := To_Tensor([
      Float_32(1.0), Float_32(0.0), Float_32(0.0),
      Float_32(0.0), Float_32(1.0), Float_32(0.0),
      Float_32(0.0), Float_32(0.0), Float_32(1.0)], [3,3]);

   Network : Del.Model.Model;

   Linear_Layer  : Del.Operators.Linear_Access_T;
   ReLU_Layer    : Del.Operators.ReLU_Access_T;
   Softmax_Layer : Del.Operators.SoftMax_Access_T;
   Loss_Function : Del.Loss.Cross_Entropy_Access_T;
   Optimizer     : Del.Optimizers.SGD_Access_T;

begin
   -- Initialize Layers
   Linear_Layer := new Del.Operators.Linear_T;
   Linear_Layer.Initialize(3, 3);
   Network.Add_Layer(Del.Func_Access_T(Linear_Layer));

   ReLU_Layer := new Del.Operators.ReLU_T;
   Network.Add_Layer(Del.Func_Access_T(ReLU_Layer));

   Softmax_Layer := new Del.Operators.SoftMax_T;
   Network.Add_Layer(Del.Func_Access_T(Softmax_Layer));

   -- Set Loss Function
   Loss_Function := new Del.Loss.Cross_Entropy_T;
   Network.Add_Loss(Del.Loss_Access_T(Loss_Function));

   -- Set Optimizer
   -- Set Optimizer
   --Optimizer := new Del.Optimizers.SGD_T'(Del.Optimizers.Create_SGD_T(0.01, 0.001, 0.9));
   --Network.Set_Optimizer(Optim_Access_T(Optimizer));  -- ✅ Type conversion

   -- Train the model
   Put_Line("Starting model training...");
   Network.Train_Model(Input, Labels, Batch_Size => 3, Num_Epochs => 10);
   Put_Line("Training complete.");

   -- Export the trained model to JSON
   --Put_Line("Exporting model...");
   --Del.Model.Export_To_JSON(Network, "model_output.json");
   --Put_Line("Model exported to JSON file.");

   -- Run forward pass on new data
   declare
      Result : Del.Tensor_T := Del.Model.Run_Layers(Network, Input);
   begin
      Put_Line("Final model output:");
      Put_Line(Result.Image);
   end;

end Main_Demo;
