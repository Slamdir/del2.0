-- File: del-model.ads
with Del.JSON; use Del.JSON;
with Del.YAML; use Del.YAML;
with Del.Data; use Del.Data;

package Del.Model is
   type Model is tagged private;

   -- Layer management
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T);
   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T);
   procedure Set_Optimizer(Self : in out Model; Optimizer : Optim_Access_T);
   
   -- Layer access
   function Get_Layer_Count(Self : Model) return Natural;
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T;
   function Get_Layers_Vector(Self : Model) return Layer_Vectors.Vector;

   -- Accuracy Access
   function Get_Accuracy(Self: Model) return Float;

   -- Data management
   procedure Set_Dataset(Self : in out Model; Dataset : Training_Data_Access);
   function Get_Dataset(Self : Model) return Training_Data_Access;
   
   -- Convenience functions to load data directly
   procedure Load_Data_From_JSON
     (Self          : in out Model;
      JSON_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T);
      
   -- New function to load data from YAML
   procedure Load_Data_From_YAML
     (Self          : in out Model;
      YAML_File     : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T);
   
   -- Generic load function that detects file type by extension
   procedure Load_Data_From_File
     (Self          : in out Model;
      Filename      : String;
      Data_Shape    : Tensor_Shape_T;
      Target_Shape  : Tensor_Shape_T);
   
   -- Model operations
   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T;
   
   -- Training procedure (single version)
   procedure Train_Model
     (Self       : in out Model;
      Batch_Size : Positive;
      Num_Epochs : Positive);

   procedure Export_ONNX
     (Self     : in Model;
      Filename : String);

   procedure Export_To_JSON
      (Self : in Model; 
      Filename : String);

private
   type Model is tagged record
      Layers    : Layer_Vectors.Vector;
      Loss_Func : Loss_Access_T;
      Optimizer : Optim_Access_T;
      Dataset   : Training_Data_Access;
      Accuracy  : Float;
   end record;
   
   -- Implementation of layer access functions
   function Get_Layer_Count(Self : Model) return Natural is
      (Natural(Self.Layers.Length));
   
   function Get_Layer(Self : Model; Index : Positive) return Func_Access_T is
      (Self.Layers.Element(Index));
      
end Del.Model;