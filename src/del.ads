with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;

with Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

package Del is

   subtype Tensor_T is Orka.Numerics.Singles.Tensors.CPU.CPU_Tensor; 
   subtype Element_T is Orka.Numerics.Singles.Tensors.Element;
   subtype Elements_T is Orka.Numerics.Singles.Tensors.Element_Array;

   package Data_Maps is new
     Ada.Containers.Indefinite_Hashed_Maps
       (Key_Type        => String,
        Element_Type    => Tensor_T,
        Hash            => Ada.Strings.Hash,
        Equivalent_Keys => "=");

   type Func_T is tagged record
      D : Data_Maps.Map;
   end record;

end Del;