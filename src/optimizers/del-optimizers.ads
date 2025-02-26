package Del.Optimizers is

   type SGD_T is new Optim_T with record
            Weight_Decay : Float;
            Momentum : Float;
   end record;
   type SGD_Access_T is access all SGD_T'Class;

   overriding procedure Step(Self : SGD_T; Layers : Layer_Vectors.Vector);
   overriding procedure Zero_Gradient (Self : SGD_T; Layers : Layer_Vectors.Vector);

   procedure Print_Stats (Self : SGD_T);

   function Create_SGD_T(Learning_Rate : Float;  Weight_Decay : Float; Momentum : Float) return SGD_T;

end Del.Optimizers;