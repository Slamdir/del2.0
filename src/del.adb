with Ada.Real_Time; use Ada.Real_Time;

package body Del is

   procedure Initialize (L : in out Func_T; In_Nodes, Out_Nodes : Positive) is
      -- Initialize with uniform random values between -0.1 and 0.1 for stable training
      Weights : Tensor_T := Random_Tensor.Normal((In_Nodes, Out_Nodes)) * 0.01;
      Bias    : Tensor_T := Zeros((1, Out_Nodes));

      Weights_Grad : Tensor_T := Zeros((In_Nodes, Out_Nodes));
      Bias_Grad : Tensor_T := Zeros((1, Out_Nodes));

      Weights_Velocity : Tensor_T := Zeros((In_Nodes, Out_Nodes));
      Bias_Velocity : Tensor_T := Zeros((1, Out_Nodes));
   begin
      L.Map.Insert("weights", Weights);
      L.Map.Insert("bias", Bias);

      L.Map.Insert("weights_grad", Weights_Grad);
      L.Map.Insert("bias_grad", Bias_Grad);

      L.Map.Insert("weights_velocity", Weights_Velocity);
      L.Map.Insert("bias_velocity", Bias_Velocity);
   end;

   SC : Seconds_Count;
   TS : Time_Span;

begin
   Ada.Real_Time.Split(Clock, SC, TS);
   Reset_Random (To_Duration(TS) * 100000);
end Del;