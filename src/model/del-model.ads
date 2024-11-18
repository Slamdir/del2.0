with Ada.Containers.Vectors;

package Del.Model is

    package Func_Vector is new 
        Ada.Containers.Vectors
         (Index_Type => Natural, 
         Element_Type => Func_Access_T);

    type Model is tagged record
        Layers : Func_Vector.Vector;
    end record;

    procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
    procedure Run_Layers(Self : in Model);

end Del.Model;