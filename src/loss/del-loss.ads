package Del.Loss is
   
   type Cross_Entropy_T is new Loss_T with null record;
   type Cross_Entropy_Access_T is access all Cross_Entropy_T'Class;

   overriding function Forward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Float;
   overriding function Backward (L : Cross_Entropy_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T;

end Del.Loss;