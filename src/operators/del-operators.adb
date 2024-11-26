with Ada.Text_IO; use Ada.Text_IO;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

package body Del.Operators is

   overriding function Forward (L : Linear_T; X : Tensor_T) return Tensor_T is
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

   overriding function Forward (L : ReLU_T; X : Tensor_T) return Tensor_T is
      Result : Tensor_T := X;
   begin
      Put_Line("Forward from ReLu_T");
      --L.map.insert {string of activated....,Tensor}
      Result := Max(X, Zeros(X.Shape));
      return Result;
   end Forward;
   
   overriding function Backward (L : ReLU_T; Dy : Tensor_T) return Tensor_T is
      -- Create a tensor of zeros same shape as gradient
      Zero : Tensor_T := Zeros(Dy.Shape);
      -- Create a tensor of ones same shape as gradient
      One : Tensor_T := Ones(Dy.Shape);
   begin
      -- Use Min to clamp the gradient to 1.0 where it should pass through
      return Min(Max(Dy, Zero), One);
      --take in result from froward to use for the backward negative propagation using L.Map{String}
   end Backward;
   
   overriding function Get_Params (L : ReLU_T) return Params_T is
      Dummy : Tensor_Access_T := null;
   begin
      return (Dummy, Dummy);
   end Get_Params;
end Del.Operators;