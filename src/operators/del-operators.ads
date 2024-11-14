package Del.Operators is

   type Func_T is abstract new Layer_T with null record;
   function Forward (Self : Func_T; X : Tensor_T) return Tensor_T is abstract;
   function Backward (Self : Func_T; X : Tensor_T) return Tensor_T is abstract;
   --function Get_Params ( Self : Func_T)  is abstract;

   type Linear_T is new Func_T with null record;
   overriding function Forward (Self : Linear_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (Self : Linear_T; X : Tensor_T) return Tensor_T;

   type ReLU_T is new Func_T with null record;
   overriding function Forward (Self : ReLU_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (Self : ReLU_T; X : Tensor_T) return Tensor_T;

end Del.Operators;