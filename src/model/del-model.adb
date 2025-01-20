with Ada.Text_IO; use Ada.Text_IO;
with Ada.Containers; use Ada.Containers;
with Ada.Exceptions;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;

package body Del.Model is
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
   begin
       Self.Layers.Append(Layer);
   end Add_Layer;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

   procedure Train_Model(Self : in Model; Num_Epochs : Positive; Data : Tensor_T) is
      Loss_Value : Element_T;
   begin
      for I in 1 .. Num_Epochs loop
         Put_Line("Epoch:" & I'Image);
         
         declare
            -- Forward pass
            Output : Tensor_T := Run_Layers(Self, Data);
         begin
            if Self.Loss_Func /= null then
               -- Compute loss and gradient
               Loss_Value := Self.Loss_Func.Forward(Data, Output);
               -- Initialize Loss_Grad with same shape as Output
               declare
                  Loss_Grad : Tensor_T := Self.Loss_Func.Backward(Data, Output);
                  Grad : Tensor_T := Loss_Grad;
                  C : Layer_Vectors.Cursor := Self.Layers.Last;
               begin
                  -- Backward pass through all layers
                  while Layer_Vectors.Has_Element(C) loop
                     declare
                        Current_Layer : constant Func_Access_T := Layer_Vectors.Element(C);
                     begin
                        Grad := Current_Layer.all.Backward(Grad);
                     end;
                     Layer_Vectors.Previous(C);
                  end loop;
               end;
               
               Put_Line("Loss:" & Loss_Value'Image);
            end if;
         end;
      end loop;
   end Train_Model;

   procedure Train_Model_From_JSON
     (Self : in Model;
      Num_Epochs : Positive;
      JSON_File : String;
      Data_Shape : Tensor_Shape_T;
      Target_Shape : Tensor_Shape_T) is
   begin
      Put_Line("Loading data from JSON file: " & JSON_File);
      
      declare
         Dataset : constant Dataset_Array := Load_Dataset(
            Filename => JSON_File,
            Data_Shape => Data_Shape,
            Target_Shape => Target_Shape);
            
         Training_Data : constant Tensor_T := Dataset(1).Data.all;
         Training_Target : constant Tensor_T := Dataset(1).Target.all;
      begin
         Put_Line("Dataset loaded successfully. Samples:" & Dataset'Length'Image);
         Put_Line("Data shape: " & Shape(Training_Data)(1)'Image & "," & 
                 Shape(Training_Data)(2)'Image);
         Put_Line("Target shape: " & Shape(Training_Target)(1)'Image & "," & 
                 Shape(Training_Target)(2)'Image);
         
         -- Call the regular Train_Model with the loaded data
         Train_Model(Self, Num_Epochs, Training_Data);
      end;
   exception
      when E : JSON_Parse_Error =>
         Put_Line("Error loading JSON data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Train_Model_From_JSON;

   function Run_Layers(Self : in Model; Input : Tensor_T) return Tensor_T is
   begin
      Put_Line("Run_Layers called with input shape: " & 
               Shape(Input)(1)'Image & "," & Shape(Input)(2)'Image);
               
      if Self.Layers.Length = 0 then
          Put_Line("No layers in network");
          return Input;
      end if;

      Put_Line("Network has" & Self.Layers.Length'Image & " layers");

      declare
         Current_Input : Tensor_T := Input;
         First_Layer  : constant Func_Access_T := Self.Layers.First_Element;
         First_Output : constant Tensor_T := First_Layer.all.Forward(Current_Input);
         Result : Tensor_T := First_Output;
         C : Layer_Vectors.Cursor := Self.Layers.First;
      begin
         -- Skip the first element since we've already processed it
         Layer_Vectors.Next(C);
         
         while Layer_Vectors.Has_Element(C) loop
             Put_Line("Processing next layer");
             Put_Line("Current input shape: " & Shape(Result)(1)'Image & "," & Shape(Result)(2)'Image);
             
             declare
                Current_Layer : constant Func_Access_T := Layer_Vectors.Element(C);
             begin
                Result := Current_Layer.all.Forward(Result);
             end;
             
             Put_Line("Layer output shape: " & Shape(Result)(1)'Image & "," & Shape(Result)(2)'Image);
             Layer_Vectors.Next(C);
         end loop;
         
         return Result;
      end;
   exception
      when E : others =>
         Put_Line("Error in Run_Layers: ");
         Put_Line(Ada.Exceptions.Exception_Information(E));
         raise;
   end Run_Layers;
end Del.Model;