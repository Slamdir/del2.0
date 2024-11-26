with Ada.Text_IO; use Ada.Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

package body Del.Operators is

   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T is  -- Added 'in out' to match spec
   begin
      Put_Line("Forward from Linear_T");
      return X;
      --(X * L.Map ("Weights") + L.Map ("Bias"));
   end Forward;
   
   overriding function Backward (L : Linear_T; Dy : Tensor_T) return Tensor_T is
   begin
      return Dy;
   end Backward;

   overriding function Get_Params (L : Linear_T) return Params_T is
      T1 : Tensor_Access_T := new Tensor_T'(Zeros((2, 2)));
      T2 : Tensor_Access_T := new Tensor_T'(Zeros((2, 2)));
   begin
      return (T1, T2);
   end Get_Params;

   overriding function Forward (L : in out ReLU_T; X : Tensor_T) return Tensor_T is
      Zero : Tensor_T := Zeros(X.Shape);
      Result : Tensor_T := Max(X, Zero);
   begin
      Put_Line("Forward from ReLu_T");
      -- Store output for backward pass
      L.Map.Include("forward_output", Result);
      return Result;
   end Forward;

   overriding function Backward (L : in ReLU_T; Dy : Tensor_T) return Tensor_T is
      Zero : Tensor_T := Zeros(Dy.Shape);
   begin
      if L.Map.Contains("forward_output") then
         declare
            Forward_Output : Tensor_T := L.Map("forward_output");
            Mask : Tensor_T := Forward_Output / (Forward_Output + Ones(Dy.Shape));
         begin
            return Dy * Mask;
         end;
      else
         return Zero;
      end if;
   end Backward;
   
   overriding function Get_Params (L : ReLU_T) return Params_T is
      Dummy : Tensor_Access_T := null;
   begin
      return (Dummy, Dummy);
   end Get_Params;
end Del.Operators;