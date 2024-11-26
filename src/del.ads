with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;

with Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

package Del is

   subtype Tensor_T is Orka.Numerics.Singles.Tensors.CPU.CPU_Tensor;
   type Tensor_Access_T is access all Tensor_T; 
   subtype Tensor_Shape_T is Orka.Numerics.Singles.Tensors.Tensor_Shape;
   subtype Element_T is Orka.Numerics.Singles.Tensors.Element;
   subtype Elements_T is Orka.Numerics.Singles.Tensors.Element_Array;
   
   package Data_Maps is new
     Ada.Containers.Indefinite_Hashed_Maps
       (Key_Type        => String,
        Element_Type    => Tensor_T,
        Hash            => Ada.Strings.Hash,
        Equivalent_Keys => "=");

   type Func_T is abstract tagged private;
   type Func_Access_T is access all Func_T'Class;

   function Forward (L : Func_T; X : Tensor_T) return Tensor_T is abstract;
   function Backward (L : Func_T; Dy : Tensor_T) return Tensor_T is abstract;

   type Loss_T is abstract tagged private;
   type Loss_Access_T is access all Loss_T'Class;

   function Forward (L : Loss_T; Expected : Tensor_T; Actual : Tensor_T) return Element_T is abstract;
   function Backward (L : Loss_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T is abstract;

private
   type Func_T is abstract tagged record
      Map : Data_Maps.Map;
   end record;

   type Loss_T is abstract tagged record
      Map : Data_Maps.Map;
   end record;
end Del;