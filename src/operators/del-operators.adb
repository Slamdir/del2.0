with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics;
with Orka.Numerics.Singles.Tensors; use Orka.Numerics.Singles.Tensors;

package body Del.Operators is
   procedure Initialize(L : in out Linear_T; In_Nodes, Out_Nodes : Positive) is
      -- Initialize with uniform random values between -0.1 and 0.1 for stable training
      Weights : Tensor_T := Random_Uniform((In_Nodes, Out_Nodes)) * 0.2 - 0.1;
      Bias    : Tensor_T := Zeros((1, Out_Nodes));
      Map : Data_Maps.Map := L.Map;
   begin
      Map.Insert("weights", Weights);
      Map.Insert("bias", Bias);
      L.Map := Map;
   end Initialize;

   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T is
         Weights : constant Tensor_T := L.Map("weights");
         Bias    : constant Tensor_T := L.Map("bias");
         Map : Data_Maps.Map := L.Map;
         Batch_Size : constant Positive := Shape(X)(1);
         Out_Size : constant Positive := Shape(Weights)(2);
         Output : Tensor_T := Zeros((Batch_Size, Out_Size));  -- Initialize with correct dimensions
      begin
         -- Store input for backward pass using Include instead of Insert
         Map.Include("input", X);
         L.Map := Map;
         
         
         for i in 1 .. Batch_Size loop
            for j in 1 .. Out_Size loop
               declare
                  Sum : Element_T := 0.0;
               begin
                  -- Dot product for this output element
                  for k in 1 .. Shape(Weights)(1) loop
                     declare
                        X_Val : constant Element_T := X.Get((i, k));
                        W_Val : constant Element_T := Weights.Get((k, j));
                     begin
                        Sum := Sum + X_Val * W_Val;
                     end;
                  end loop;
                  -- Add bias
                  Sum := Sum + Bias.Get((1, j));
                  -- Set output
                  Output.Set((i, j), Sum);
               end;
            end loop;
         end loop;
         
         return Output;
      end Forward;

   overriding function Backward (L : in out Linear_T; Dy : Tensor_T) return Tensor_T is
      Input   : constant Tensor_T := L.Map("input");
      Weights : constant Tensor_T := L.Map("weights");
      Batch_Size : constant Positive := Shape(Input)(1);
      
      -- Get current gradients or initialize if they don't exist
      Weights_Grad : Tensor_T := (if L.Map.Contains("weights_grad") 
                                 then L.Map("weights_grad") 
                                 else Zeros(Shape(Weights)));
      Bias_Grad    : Tensor_T := (if L.Map.Contains("bias_grad") 
                                 then L.Map("bias_grad") 
                                 else Zeros((1, Shape(Weights)(2))));
      Map : Data_Maps.Map := L.Map;
      
      -- For computing bias gradients
      New_Bias_Grad : Tensor_T := Zeros((1, Shape(Dy)(2)));
      Sum_Row : Tensor_T := Dy(1);  -- Initialize with first row
   begin
      -- Update gradients
      -- weights_grad = input.T * dy
      Weights_Grad := Add(Weights_Grad, Multiply(Transpose(Input), Dy));
      
      -- bias_grad = sum(dy, axis=0)
      -- Sum all rows
      for I in 2 .. Batch_Size loop
         Sum_Row := Add(Sum_Row, Dy(I));
      end loop;
      -- Set as first (and only) row of New_Bias_Grad
      New_Bias_Grad.Set(1, Sum_Row);
      
      Bias_Grad := Add(Bias_Grad, New_Bias_Grad);
      
      -- Store updated gradients
      Map.Insert("weights_grad", Weights_Grad);
      Map.Insert("bias_grad", Bias_Grad);
      L.Map := Map;
      
      -- Return gradient with respect to input
      -- grad_input = dy * weights.T
      return Multiply(Dy, Transpose(Weights));
   end Backward;

   overriding function Get_Params (L : Linear_T) return Params_T is
      Weights : Tensor_Access_T := new Tensor_T'(L.Map("weights"));
      Bias    : Tensor_Access_T := new Tensor_T'(L.Map("bias"));
   begin
      return (0 => Weights, 1 => Bias);
   end Get_Params;

   overriding function Forward (L : in out ReLU_T; X : Tensor_T) return Tensor_T is
      Zero : Tensor_T := Zeros(X.Shape);
      Result : Tensor_T := Max(X, Zero);
   begin
      Put_Line("Forward from ReLu_T");
      -- Store output for backward pass
      L.Map.Include("forward_output", Result);
      return Result;
   end Forward;

   overriding function Backward (L : in out ReLU_T; Dy : Tensor_T) return Tensor_T is
      Zero : Tensor_T := Zeros(Dy.Shape);
      Map : Data_Maps.Map := L.Map;
   begin
      if Map.Contains("forward_output") then
         declare
            Forward_Output : Tensor_T := Map("forward_output");
            Mask : Tensor_T := Forward_Output / (Forward_Output + Ones(Dy.Shape));
         begin
            return Dy * Mask;
         end;
      else
         return Zero;
      end if;
   end Backward;
   
   overriding function Get_Params (L : ReLU_T) return Params_T is
      Dummy : Tensor_Access_T := null;
   begin
      return (Dummy, Dummy);
   end Get_Params;

   function Row_Sum(Values : Tensor_T) return Tensor_T is
      Rows     : Integer := Shape(Values)(1);
      Output   : Tensor_T := Zeros((Rows, 1));
   begin
      --  Put_Line ("Rows: " & Rows'Image & " Columns: " & Columns'Image);
      for I in 1 .. Rows loop
      declare
         Row_I : Tensor_T := Values(I);
         begin
            Output.Set(I, Sum(Row_I));
         end;
      end loop;
      --  Put_Line(Image(Output));
      return Output;
   end Row_Sum;


   -- Allows for Cross_Entropy to call SoftMax with the Actual Values
   function SoftMax(X : Tensor_T) return Tensor_T is

      function Divide_By_Row (Exp_Values : Tensor_T; Exp_Values_Sum : Tensor_T) return Tensor_T is
         Rows     : Integer := Shape(Exp_Values)(1);
         Output   : Tensor_T := Zeros(Shape(Exp_Values));
      begin
         for I in 1 .. Rows loop
         declare
            Exp_Values_Row_I     : Tensor_T := Exp_Values(I);
            Exp_Values_Element_I : Element_T := Exp_Values_Sum(I);
            begin
               Output.Set(I, Exp_Values_Row_I / Exp_Values_Element_I);
            end;
         end loop;
         return Output;
      end Divide_By_Row;

      Exp_Values     : Tensor_T := Ada.Numerics.e ** X;
      Exp_Values_Sum : Tensor_T := Row_Sum(Exp_Values);
      Output         : Tensor_T := Divide_By_Row(Exp_Values, Exp_Values_Sum);

   begin
      return Output;
   end SoftMax;

   -- Acts as a proxy to call SoftMax
   overriding function Forward (L : in out SoftMax_T; X : Tensor_T) return Tensor_T is
      Output : Tensor_T := Softmax(X);
   begin
      return Output;
   end Forward;

   overriding function Backward (L : in out SoftMax_T; Dy : Tensor_T) return Tensor_T is
   begin
      return Dy;  -- Your existing implementation
   end Backward;

   overriding function Get_Params (L : SoftMax_T) return Params_T is
   Dummy : Tensor_Access_T := null;
   begin
      return (Dummy, Dummy);
   end Get_Params;

end Del.Operators;
