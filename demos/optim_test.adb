with Del.Optimizers;

procedure Optim_Test is

    SGD : SGD_T;

begin

    SGD := SGD_T_Package.Create_SGD_T(0.01, 0.0001, 0.9);
    SGD.Print_Stats;
    
end Optim_Test;