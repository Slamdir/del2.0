with Ada.Containers.Vectors;
with Orka.Numerics.Singles.Tensors;
with Del.JSON; use Del.JSON;  -- Add this line

package Del.Model is
   type Model is tagged private;

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);

   -- Updated Train_Model to support both direct tensor and JSON input
   procedure Train_Model
     (Self : in Model;
      Num_Epochs : Positive;
      Data : Tensor_T);
      
   -- New procedure for training from JSON file
procedure Train_Model_From_JSON
     (Self : in Model;
      Num_Epochs : Positive;
      JSON_File : String;
      Data_Shape : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T);  

   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);

private
   package Layer_Vectors is new
     Ada.Containers.Vectors
       (Index_Type   => Positive,
        Element_Type => Func_Access_T);

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
   end record;
end Del.Model;