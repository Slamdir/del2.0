package body Del.Optimizers is

    package body SGD_T_Package is
        function Create_SGD_T(
            Learning_Rate : Float; 
            Weight_Decay : Float; 
            Momentum : Float;
            Object : SGD_T) return SGD_T is
        begin
            Object.Parameters := Params;
            Object.Learning_Rate := Learning_Rate;
            Object.Weight_Decay := Weight_Decay;
            Object.Momentum := Momentum;
            return Object;
        end Create_SGD_T;

    end SGD_T_Package;

    procedure Print_Stats(Self : Optim_T) is

    begin
        Put_Line("Learning Rate: " & Self.Learning_Rate'Image);
        Put_Line("Weight Decay: " & Self.Weight_Decay'Image);
    end Print_Stats;

end Del.Optimizers;