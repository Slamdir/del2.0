with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;
with Ada.Containers.Vectors;
with Ada.Containers.Indefinite_Vectors;
with Orka; use Orka;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;
with Ada.Text_IO; use Ada.Text_IO;

package Del is

   subtype Tensor_T is Orka.Numerics.Singles.Tensors.CPU.CPU_Tensor;
   type Tensor_Access_T is access all Tensor_T; 
   subtype Tensor_Shape_T is Orka.Numerics.Singles.Tensors.Tensor_Shape;
   subtype Element_T is Orka.Numerics.Singles.Tensors.Element;
   subtype Elements_T is Orka.Numerics.Singles.Tensors.Element_Array;

   package Random_Tensor is new Generic_Random(Tensor_T);

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

   function Forward (L : in out Func_T; X : Tensor_T) return Tensor_T is abstract;
   function Backward (L : in out Func_T; Dy : Tensor_T) return Tensor_T is abstract;
   function Get_Params (L : Func_T) return Params_T is abstract;
   procedure Initialize (L : in out Func_T; In_Nodes, Out_Nodes : Positive);

   type Loss_T is abstract tagged private;
   type Loss_Access_T is access all Loss_T'Class;

   function Forward  (L : Loss_T; Expected : Tensor_T; Actual : Tensor_T) return Float is abstract;
   function Backward (L : Loss_T; Expected : Tensor_T; Actual : Tensor_T) return Tensor_T is abstract;

   type Optim_T is abstract tagged private;
   type Optim_Access_T is access all Optim_T'Class;

   --Vector of layers (needed for optim and model)
   package Layer_Vectors is new
     Ada.Containers.Vectors
       (Index_Type   => Positive,
        Element_Type => Func_Access_T);

   procedure Step (Self : Optim_T; Layers : Layer_Vectors.Vector) is abstract;
   procedure Zero_Gradient (Self : Optim_T; Layers : Layer_Vectors.Vector) is abstract;

private
   type Func_T is abstract tagged record
      Map : Data_Maps.Map;
   end record;

   type Loss_T is abstract tagged record
      Map : Data_Maps.Map;
   end record;

   type Optim_T is abstract tagged record
      Learning_Rate : Float;
   end record;

end Del;