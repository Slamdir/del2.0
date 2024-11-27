with Ada.Containers.Vectors;
with Orka.Numerics.Singles.Tensors;

package Del.Model is
   type Model is tagged private;

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;

private
   -- Vector to store layers
   package Layer_Vectors is new
     Ada.Containers.Vectors
       (Index_Type   => Positive,
        Element_Type => Func_Access_T);

   type Model is tagged record
      Layers : Layer_Vectors.Vector;
   end record;
end Del.Model;