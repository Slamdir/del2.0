with Ada.Containers.Vectors;
with Orka.Numerics.Singles.Tensors;
with Del.JSON; use Del.JSON;

package Del.Model is
   type Model is tagged private;

   -- Layer management
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);
   
   -- Layer access
   function Get_Layer_Count(Self : Model) return Natural;
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T;

   -- Model operations
   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;
   
   -- Training procedure
   procedure Train_Model(
      Self       : in Model;
      Num_Epochs : Positive;
      Data       : Tensor_T;
      Labels     : Tensor_T;
      JSON_File  : String := "";
      JSON_Data_Shape   : Tensor_Shape_T := (1 => 1, 2 => 1);
      JSON_Target_Shape : Tensor_Shape_T := (1 => 1, 2 => 1));

   -- ONNX export
   procedure Export_ONNX(
      Self : in Model;
      Filename : String);

private
   package Layer_Vectors is new
     Ada.Containers.Vectors
       (Index_Type   => Positive,
        Element_Type => Func_Access_T);

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
   end record;
   
   -- Make Layer_Vectors visible to ONNX package
   function Get_Layers_Vector(Self : Model) return Layer_Vectors.Vector;
   
   -- Implementation of layer access functions
   function Get_Layer_Count(Self : Model) return Natural is
      (Natural(Self.Layers.Length));
   
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T is
      (Self.Layers.Element(Index));
end Del.Model;