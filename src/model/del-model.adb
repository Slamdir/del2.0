with Ada.Containers; use Ada.Containers;
with Ada.Exceptions;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;
with Ada.Numerics.Float_Random;

package body Del.Model is
   procedure Add_Layer(Self : in out Model; Layer : Func_Access_T) is
   begin
       Self.Layers.Append(Layer);
   end Add_Layer;

   procedure Add_Loss(Self : in out Model; Loss_Func : Loss_Access_T) is 
   begin 
      Self.Loss_Func := Loss_Func;
   end Add_Loss;

 procedure Train_Model
     (Self       : in Model;
      Num_Epochs : Positive;
      Data       : Tensor_T;
      Labels     : Tensor_T;
      JSON_File  : String := "";
      JSON_Data_Shape   : Tensor_Shape_T := (1 => 1, 2 => 1);
      JSON_Target_Shape : Tensor_Shape_T := (1 => 1, 2 => 1))
   is
      type Index_Array is array (Positive range <>) of Integer;
      Indecies        : Index_Array(1..Shape(Data)(1));
      Training_Data   : Tensor_T := Data;
      Training_Labels : Tensor_T := Labels;
      Loss_Value : Element_T;
   begin
      -- If JSON file is provided, load data from it
      if JSON_File /= "" then
         Put_Line("Loading data from JSON file: " & JSON_File);
         declare
            Dataset : constant Dataset_Array := Load_Dataset(
               Filename => JSON_File,
               Data_Shape => JSON_Data_Shape,
               Target_Shape => JSON_Target_Shape);
         begin
            Training_Data := Dataset(1).Data.all;
            Training_Labels := Dataset(1).Target.all;
            Put_Line("Dataset loaded successfully. Samples:" & Dataset'Length'Image);
         end;
      end if;

      -- self: item calling the function?
      -- data: input node values
      -- target: target label (same size as input node values)
      -- batch_size: size of batches input node data is handled in
      -- num_epochs: number of repitions on the passed dataset (should be kept at 1 or a low number to avoid overfit)
      -- optimizer: object that handles gradient decent
      -- loss_fn: object that handles loss function
      -- def fit(self,data,target,batch_size,num_epochs,optimizer,loss_fn):

      for epoch in 1 .. Num_Epochs loop
      --Just to have this compile
      Put_Line("");
            -- shuffle Indecies


            -- shuffle -- generate a complete and distinct set of index values the same size as the dataset and store in shuffle object
            -- loop accross number of batches in data (last one may be incomplete)
                -- reset optimizer internal values for new loop
                -- feedforward next batchsize of data (loop)
                -- find average loss
                -- loss := loss_fn.forward(X,Y) -- average loss
                -- grad := loss_fn.backward() -- initial gradient
                -- loop to compute remaining gradients
                -- optimizer.step()
                     -- apply gradient changes accross all weights and biases
        -- return loss_history
      end loop;

      for I in 1 .. Num_Epochs loop
         Put_Line("Epoch:" & I'Image);
         
         declare
            -- Forward pass
            Output : Tensor_T := Run_Layers(Self, Training_Data);
         begin
            if Self.Loss_Func /= null then
               -- Compute loss and gradient
               Loss_Value := Self.Loss_Func.Forward(Training_Labels, Output);
               
               declare
                  Loss_Grad : Tensor_T := Self.Loss_Func.Backward(Training_Labels, Output);
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
   exception
      when E : JSON_Parse_Error =>
         Put_Line("Error loading JSON data: " & Ada.Exceptions.Exception_Message(E));
         raise;
      when E : others =>
         Put_Line("Unexpected error: " & Ada.Exceptions.Exception_Message(E));
         raise;
   end Train_Model;

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

function Get_Params(Self : Model) return Layer_Vectors.Vector is
begin
   return Self.Layers;
end Get_Params;

end Del.Model;