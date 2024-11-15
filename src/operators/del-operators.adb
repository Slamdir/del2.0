with Ada.Text_IO; use Ada.Text_IO;

with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

package body Del.Operators is

   overriding function Forward (L : Linear_T; X : Tensor_T) return Tensor_T is
      (X ** L.Map ("Weights") + L.Map ("Bias"));
   
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
   begin
      Put_Line (L.Map'Image & " Forward from ReLu_T");
      return X;
   end Forward;

   overriding function Backward (L : ReLU_T; Dy : Tensor_T) return Tensor_T is
   begin
      return dY;
   end Backward;

   overriding function Get_Params (L : ReLU_T) return Params_T is
      T1 : Tensor_Access_T := new Tensor_T'(Zeros((2, 2)));
      T2 : Tensor_Access_T := new Tensor_T'(Zeros((2, 2)));
   begin
      return (T1, T2);
   end Get_Params;

end Del.Operators;