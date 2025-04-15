package Del.Operators is
   type Linear_T is new Func_T with null record;
   type Linear_Access_T is access all Linear_T'Class;
   
   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (L : in out Linear_T; Dy : Tensor_T) return Tensor_T;
   overriding function Get_Params (L : Linear_T) return Params_T;

   type ReLU_T is new Func_T with null record;
   type ReLU_Access_T is access all ReLU_T'Class;

   overriding function Forward (L : in out ReLU_T; X : Tensor_T) return Tensor_T;  
   overriding function Backward (L : in out ReLU_T; Dy : Tensor_T) return Tensor_T;    
   overriding function Get_Params (L : ReLU_T) return Params_T;

   type SoftMax_T is new Func_T with null record;
   type SoftMax_Access_T is access all SoftMax_T'Class;
   
   overriding function Forward (L : in out SoftMax_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (L : in out SoftMax_T; Dy : Tensor_T) return Tensor_T;
   overriding function Get_Params (L : SoftMax_T) return Params_T;

   type Sigmoid_T is new Func_T with null record;
   type Sigmoid_Access_T is access all Sigmoid_T'Class;

   overriding function Forward (L : in out Sigmoid_T; X : Tensor_T) return Tensor_T;
   overriding function Backward (L : in out Sigmoid_T; Dy : Tensor_T) return Tensor_T;
   overriding function Get_Params (L : Sigmoid_T) return Params_T;

   function SoftMax(X : Tensor_T) return Tensor_T;

end Del.Operators;