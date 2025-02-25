with Del.JSON; use Del.JSON;

package Del.Model is

   type Model is tagged private;

   procedure Add_Layer (Self : in out Model; Layer : Func_Access_T);

   -- Training procedure with tensor inputs
   procedure Train_Model
     (Self       : in Model;
      Data       : Tensor_T;
      Labels     : Tensor_T;
      Batch_Size : Positive;
      Num_Epochs : Positive);

   -- Training procedure with JSON file input
   procedure Train_Model_JSON
     (Self          : in out Model;
      JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T;
      Batch_Size    : Positive;
      Num_Epochs    : Positive);

   function Run_Layers (Self : in Model; Input : Tensor_T) return Tensor_T;

   procedure Add_Loss (Self : in out Model; Loss_Func : Loss_Access_T);

   function Get_Params (Self : Model) return Layer_Vectors.Vector;

private

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
      Optimizer : Optim_Access_T;
   end record;

end Del.Model;