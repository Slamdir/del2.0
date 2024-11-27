package Del.Operators is
   type Linear_T is new Func_T with null record;
   type Linear_Access_T is access all Linear_T'Class;

   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (L : in Linear_T; Dy : Tensor_T) return Tensor_T;
   overriding function Get_Params (L : Linear_T) return Params_T;

   type ReLU_T is new Func_T with null record;

   overriding function Forward (L : in out ReLU_T; X : Tensor_T) return Tensor_T;  
   overriding function Backward (L : in ReLU_T; Dy : Tensor_T) return Tensor_T;    
   overriding function Get_Params (L : ReLU_T) return Params_T;

end Del.Operators;