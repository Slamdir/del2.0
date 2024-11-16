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

   type Index_T is range 0 .. 1;
   type Params_T is array (Index_T) of Tensor_Access_T;

   type Func_T is abstract tagged private;
   type Func_Access_T is access all Func_T'Class;
   type Funcs_T is array (1 .. 2) of Func_Access_T;

   function Forward (L : Func_T; X : Tensor_T) return Tensor_T is abstract;
   function Backward (L : Func_T; Dy : Tensor_T) return Tensor_T is abstract;
   function Get_Params (E : Func_T) return Params_T is abstract;

private
   type Func_T is abstract tagged record
      Map : Data_Maps.Map;
   end record;
end Del;