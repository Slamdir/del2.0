-- File: del-model.ads
with Del.JSON; use Del.JSON;

package Del.Model is

   type Training_Data is tagged private;
   type Training_Data_Access is access all Training_Data'Class;

   type Model is tagged private;

   -- Layer management
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);
   
   -- Layer access
   function Get_Layer_Count(Self : Model) return Natural;
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T;
   function Get_Layers_Vector(Self : Model) return Layer_Vectors.Vector;

   -- Data management
   procedure Load_Data_From_JSON
     (Self          : in out Model;
      JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T);
      
   procedure Set_Data
     (Self   : in out Model;
      Data   : Tensor_T;
      Labels : Tensor_T);
      
   function Get_Data(Self : Model) return Tensor_T;
   function Get_Labels(Self : Model) return Tensor_T;
   
   -- Model operations
   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;
   
   -- Training procedures
   procedure Train_Model
     (Self       : in out Model;
      Batch_Size : Positive;
      Num_Epochs : Positive);
      
   procedure Train_Model
     (Self       : in Model;
      Data       : Tensor_T;
      Labels     : Tensor_T;
      Batch_Size : Positive;
      Num_Epochs : Positive);

   procedure Export_ONNX
     (Self     : in Model;
      Filename : String);

private
   type Training_Data is tagged record
      Data   : Tensor_Access_T;
      Labels : Tensor_Access_T;
   end record;

   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
      Optimizer : Optim_Access_T;
      Dataset   : Training_Data_Access;
   end record;
   
   -- Implementation of layer access functions
   function Get_Layer_Count(Self : Model) return Natural is
      (Natural(Self.Layers.Length));
   
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T is
      (Self.Layers.Element(Index));
      
end Del.Model;