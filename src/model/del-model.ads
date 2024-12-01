with Ada.Containers.Vectors;
with Orka.Numerics.Singles.Tensors;

package Del.Model is
   type Model is tagged private;

   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);

   --This is our fit:
      -- Data, Target, and Batch_Size are used in Data Generation
      -- Optimizer and Loss Function are added into the model itself (just calls them when needed)
      -- Num_Epochs is number of loops for training
   procedure Train_Model(Self : in Model; Num_Epochs : Positive; Data : Tensor_T);

   --This is our predict
   function  Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);
   --  procedure Add_Optim(Self : in out Model; Loss_Func : Loss_Access_T);

private
   -- Vector to store layers
   package Layer_Vectors is new
     Ada.Containers.Vectors
       (Index_Type   => Positive,
        Element_Type => Func_Access_T);

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
      --  Optimizer : Optimizer_T (or whatever we call it)
   end record;
end Del.Model;