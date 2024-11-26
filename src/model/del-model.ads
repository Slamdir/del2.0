with Ada.Containers.Vectors;

package Del.Model is

    package Func_Vector is new 
        Ada.Containers.Vectors
         (Index_Type => Natural, 
         Element_Type => Func_Access_T);

    type Model is tagged record
        Layers    : Func_Vector.Vector;
        Loss_Func : Loss_Access_T;
        -- Optimizer : Optimizer_Access_T (Or whatever we call it)
    end record;

    procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
    procedure Run_Layers(Self : in Model);

    procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);

    -- procedure Add_Optimizer(Self : in out Model; Optim : Optimizer_Access_T);

end Del.Model;