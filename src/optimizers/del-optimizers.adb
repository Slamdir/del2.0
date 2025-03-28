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
      Grad_Clip_Max : constant Element_T := 1.0;
      Grad_Clip_Min : constant Element_T := -1.0;
      Weight_Clip_Max : constant Element_T := 1.0E16;
      Weight_Clip_Min : constant Element_T := -1.0E16;
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
                  LR : constant Element_T := Element_T(Self.Learning_Rate);
                  WD : constant Element_T := Element_T(Self.Weight_Decay);
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
                  Layer_Vectors.Element(C).Map.Insert("weights_velocity", Layer_Weight_Velocity);
                  Layer_Vectors.Element(C).Map.Insert("weights", Layer_Weight_Data);
                  Layer_Vectors.Element(C).Map.Insert("bias_velocity", Layer_Bias_Velocity);
                  Layer_Vectors.Element(C).Map.Insert("bias", Layer_Bias_Data);
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
               Layer_Weight_Grad     : Tensor_T := Layer_Vectors.Element(C).Map("weights_grad");
               Layer_Bias_Grad       : Tensor_T := Layer_Vectors.Element(C).Map("bias_grad");

               Weight_Shape     : Tensor_Shape_T := Layer_Vectors.Element(C).Map("weights_grad").Shape;
               Bias_Shape       : Tensor_Shape_T := Layer_Vectors.Element(C).Map("bias_grad").Shape;
            begin

               Put_Line("Layer Weight Before Zero Grad");
               Put_line(Layer_Weight_Grad.Image);
               New_Line;

               Put_Line("Layer Bias Before Zero Grad");
               Put_Line(Layer_Bias_Grad.Image);
               New_Line;

               --Zero Weight Grad
                  Layer_Vectors.Element(C).Map("weights_grad") := Zeros(Weight_Shape);
               --Zero Bias Grad
                  Layer_Vectors.Element(C).Map("bias_grad") := Zeros(Bias_Shape);

                  declare 
                     Layer_Weight_Grad2     : Tensor_T := Layer_Vectors.Element(C).Map("weights_grad");
                     Layer_Bias_Grad2      : Tensor_T := Layer_Vectors.Element(C).Map("bias_grad");
                  begin
                     Put_Line("Layer Weight After Zero Grad");
                     Put_line(Layer_Weight_Grad2.Image);
                     New_Line;

                     Put_Line("Layer Bias After Zero Grad");
                     Put_Line(Layer_Bias_Grad2.Image);
                     New_Line;
                  end;
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