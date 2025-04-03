package body Del.Optimizers is

   function Create_SGD_T(Learning_Rate : Float;  Weight_Decay : Float; Momentum : Float) return SGD_T is
      Object : SGD_T;
   begin
      Object.Learning_Rate := Learning_Rate;
      Object.Weight_Decay := Weight_Decay;
      Object.Momentum := Momentum;
      return Object;
   end Create_SGD_T;

   overriding procedure Step(Self : SGD_T; Layers : Layer_Vectors.Vector) is
      C : Layer_Vectors.Cursor := Layers.First;
   begin
      while Layer_Vectors.Has_Element(C) loop
         if Layer_Vectors.Element(C).Map.Contains("weights") then
               declare
                  -- Parameter retrieval
                  Layer_Weight_Data     : Tensor_T := Layer_Vectors.Element(C).Map("weights");
                  Layer_Bias_Data       : Tensor_T := Layer_Vectors.Element(C).Map("bias");
                  Layer_Weight_Grad     : Tensor_T := Layer_Vectors.Element(C).Map("weights_grad");
                  Layer_Bias_Grad       : Tensor_T := Layer_Vectors.Element(C).Map("bias_grad");
                  Layer_Weight_Velocity : Tensor_T := Layer_Vectors.Element(C).Map("weights_velocity");
                  Layer_Bias_Velocity   : Tensor_T := Layer_Vectors.Element(C).Map("bias_velocity");

                  -- Local parameters
                  LR       : constant Element_T := Element_T(Self.Learning_Rate);
                  WD       : constant Element_T := Element_T(Self.Weight_Decay);
                  Momentum : constant Element_T := Element_T(Self.Momentum);
               begin
                  -- Weight update with corrected momentum/decay
                  Layer_Weight_Velocity := Momentum * Layer_Weight_Velocity 
                                          + (Layer_Weight_Grad + WD * Layer_Weight_Data);
                  Layer_Weight_Data := Layer_Weight_Data - LR * Layer_Weight_Velocity;

                  -- Bias update (no weight decay)
                  Layer_Bias_Velocity := Momentum * Layer_Bias_Velocity 
                                       + Layer_Bias_Grad;
                  Layer_Bias_Data := Layer_Bias_Data - LR * Layer_Bias_Velocity;

                  -- Store updated parameters
                  Layer_Vectors.Element(C).Map.Include("weights_velocity", Layer_Weight_Velocity);
                  Layer_Vectors.Element(C).Map.Include("weights", Layer_Weight_Data);
                  Layer_Vectors.Element(C).Map.Include("bias_velocity", Layer_Bias_Velocity);
                  Layer_Vectors.Element(C).Map.Include("bias", Layer_Bias_Data);
               end;
         end if;
         Layer_Vectors.Next(C);
      end loop;
   end Step;


   overriding procedure Zero_Gradient(Self : SGD_T; Layers : Layer_Vectors.Vector) is 
      C : Layer_Vectors.Cursor := Layers.First;
   begin

      while Layer_Vectors.Has_Element(C) loop
         if Layer_Vectors.Element(C).Map.Contains("weights") then
            declare
               Weight_Shape     : Tensor_Shape_T := Layer_Vectors.Element(C).Map("weights_grad").Shape;
               Bias_Shape       : Tensor_Shape_T := Layer_Vectors.Element(C).Map("bias_grad").Shape;
            begin
               --Zero Weight Grad
               Layer_Vectors.Element(C).Map("weights_grad") := Zeros(Weight_Shape);
               --Zero Bias Grad
               Layer_Vectors.Element(C).Map("bias_grad") := Zeros(Bias_Shape);
            end;
         end if;
         Layer_Vectors.Next(C);
      end loop;
   end Zero_Gradient;

   procedure Print_Stats(Self : SGD_T) is

    begin
        Put_Line("Learning Rate: " & Self.Learning_Rate'Image);
        Put_Line("Weight Decay: " & Self.Weight_Decay'Image);
        Put_Line("Momentum: " & Self.Momentum'Image);
    end Print_Stats;

end Del.Optimizers;