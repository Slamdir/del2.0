with Del.JSON; use Del.JSON;

package Del.Model is

   type Model is tagged private;

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);

   -- Unified training procedure that handles both tensor and JSON input
   procedure Train_Model
     (Self       : in Model;
      Num_Epochs : Positive;
      Data       : Tensor_T;
      Labels     : Tensor_T;
      JSON_File  : String := "";           -- Optional JSON file
      JSON_Data_Shape   : Tensor_Shape_T := (1 => 1, 2 => 1);  -- Shape for JSON data
      JSON_Target_Shape : Tensor_Shape_T := (1 => 1, 2 => 1)); -- Shape for JSON targets

   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);

   function Get_Params(Self : Model) return Layer_Vectors.Vector;

private

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
      Optimizer : Optim_Access_T;
   end record;

end Del.Model;