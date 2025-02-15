package Del.Optimizers is

    procedure Print_Stats (Self : Optim_T);

    private
        type SGD_T is new Optim_T with record
                Weight_Decay : Float;
                Momentum : Float;
                Velocity : Tensor_Vector.Vector;
        end record;

        type SGD_Access_T is access all SGD_T'Class;

        function Create_SGD_T(
            Params : Vector_Vector.Vector; 
            Learning_Rate : Float; 
            Weight_Decay : Float; 
            Momentum : Float;
            Object : SGD_T) return SGD_T;

end Del.Optimizers;