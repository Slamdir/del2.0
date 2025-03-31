with Ada.Exceptions;
with Ada.Numerics;
with Ada.Numerics.Elementary_Functions;
package body Del.Operators is

   function Col_Sum(Values : Tensor_T) return Tensor_T is
      Rows     : Integer := Shape(Values)(1);
      Cols     : Integer := Shape(Values)(2);
      Output   : Tensor_T := Zeros((Rows, 1));
   begin
      for I in 1 .. Rows loop
         declare
            Row_I   : Tensor_T := Values(I);
            Sum     : Element_T := 0.0;
            begin
               for J in 1 .. Cols loop
                  declare
                     Temp : Element_T := Row_I(J);
                  begin
                     Sum := Sum + Temp;
                  end;
               end loop;
               Output.Set(I, Sum);
            end;
      end loop;
      --  Put_Line(Image(Output));
      return Output;
   end Col_Sum;

   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T is
   begin
      Put_Line("Linear_T.Forward - Input shape: " & 
               Shape(X)(1)'Image & "," & Shape(X)(2)'Image);
      
      declare
         Weights : constant Tensor_T := L.Map("weights");
         Bias    : constant Tensor_T := L.Map("bias");
      begin
         Put_Line("Input shape: " & Shape(X)(1)'Image & "," & Shape(X)(2)'Image);
         Put_Line("Weights shape: " & Shape(Weights)(1)'Image & "," & Shape(Weights)(2)'Image);
         -- Rest of the code
                  
         -- Perform matrix multiplication
         declare
            Product : constant Tensor_T := X * Weights;
            
            -- Create a result tensor for adding bias
            Result : Tensor_T := Zeros(Product.Shape);
            
            -- For each row in the product
            Batch_Size : constant Positive := Shape(Product)(1);
            Features : constant Positive := Shape(Product)(2);
         begin
            Put_Line("Matrix multiplication successful");
            Put_Line("Product shape: " & 
                     Shape(Product)(1)'Image & "," & Shape(Product)(2)'Image);
            
            -- For each row in the result, add the bias (first row of Bias tensor)
            for I in 1 .. Batch_Size loop
               Result.Set(I, Add(Product(I), Bias(1)));
            end loop;

            Put_Line("Bias addition successful");
            Put_Line("Result shape: " & 
                     Shape(Result)(1)'Image & "," & Shape(Result)(2)'Image);
            
            -- Store input for backward pass
            L.Map.Include("input", X);
            
            -- Return the result
            return Result;
         end;
      end;
   exception
      when E : others =>
         Put_Line("Error in Linear_T.Forward: " & 
                  Ada.Exceptions.Exception_Message(E));
         raise;
   end Forward;

   overriding function Backward (L : in out Linear_T; Dy : Tensor_T) return Tensor_T is
      Input      : constant Tensor_T := L.Map("input");
      Weights    : constant Tensor_T := L.Map("weights");
      Batch_Size : constant Positive := Shape(Input)(1);

      -- Get current gradients
      Weights_Grad   : Tensor_T := L.Map("weights_grad");
      Bias_Grad      : Tensor_T := L.Map("bias_grad");

      -- sum(d_y, axis = 0)
      New_Bias_Grad : Tensor_T := Col_Sum(Dy.Transpose).Transpose;
   begin
      -- Update gradients
      -- weights_grad = input.T * dy
      Weights_Grad := Add(Weights_Grad, Multiply(Transpose(Input), Dy));

      -- bias_grad = bias_grad + sum(d_y, axis = 0) 
      Bias_Grad := Add(Bias_Grad, New_Bias_Grad);
      
      -- Store updated gradients
      L.Map.Include("weights_grad", Weights_Grad);
      L.Map.Include("bias_grad", Bias_Grad);
      
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
      Map  : Data_Maps.Map := L.Map;
   begin
      if Map.Contains("forward_output") then
         declare
            Forward_Output : Tensor_T := Map("forward_output");
            Mask : Tensor_T := Forward_Output > Zeros(Forward_Output.Shape);
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

   function Find_Row_Max(T : Tensor_T) return Tensor_T is
      Rows : constant Positive := Shape(T)(1);
      Cols : constant Positive := Shape(T)(2);
      Result : Tensor_T := Zeros((Rows, 1));
   begin
      for I in 1 .. Rows loop
         -- Start with first element of row as max
         declare
            Max_Val : Element_T := T.Get((I, 1));
         begin
            -- Find maximum value in row
            for J in 2 .. Cols loop
               declare
                  Current_Val : constant Element_T := T.Get((I, J));
               begin
                  if Current_Val > Max_Val then
                     Max_Val := Current_Val;
                  end if;
               end;
            end loop;
            Result.Set(I, Max_Val);
         end;
      end loop;
      return Result;
   end Find_Row_Max;

   -- Allows for Cross_Entropy to call SoftMax with the Actual Values
   function SoftMax(X : Tensor_T) return Tensor_T is
   
   -- Main SoftMax implementation with numerical stability
   Rows : constant Positive := Shape(X)(1);
   Cols : constant Positive := Shape(X)(2);
   
   -- Find max values for each row
   Max_Values : constant Tensor_T := Find_Row_Max(X);
   
   -- Create shifted input 
   Shifted_X : Tensor_T := Zeros(X.Shape);
   begin
   -- Subtract max value from each element in the row
      for I in 1 .. Rows loop
         for J in 1 .. Cols loop
            declare
               Current_Value : constant Element_T := X.Get((I, J));
               Max_Value : constant Element_T := Max_Values.Get(I);
               New_Value : constant Element_T := Current_Value - Max_Value;
            begin
               Shifted_X.Set((I, J), New_Value);
            end;
         end loop;
      end loop;

   declare
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

      Exp_Values     : Tensor_T := Exp( Shifted_X );
      Exp_Values_Sum : Tensor_T := Col_Sum(Exp_Values);
      Output         : Tensor_T := Divide_By_Row(Exp_Values, Exp_Values_Sum);

      begin
         return Output;
      end;
   end SoftMax;

   -- Acts as a proxy to call SoftMax
   overriding function Forward (L : in out SoftMax_T; X : Tensor_T) return Tensor_T is
      Output : Tensor_T := Softmax(X);
   begin
      return Output;
   end Forward;

   overriding function Backward (L : in out SoftMax_T; Dy : Tensor_T) return Tensor_T is

      Flat     : Tensor_T := Dy.Flatten;
      Diag     : Tensor_T := Diagonal(Flat);
      Off_Diag : Tensor_T := Outer(Flat, Flat);

   begin

      --  return Diag - Off_Diag;
      return Dy;

   end Backward;

   overriding function Get_Params (L : SoftMax_T) return Params_T is
   Dummy : Tensor_Access_T := null;
   begin
      return (Dummy, Dummy);
   end Get_Params;

   --  TZ   : Time_Offset := UTC_Time_Offset;
   --  Zero : Ada.Calendar.Time        :=
   --    Ada.Calendar.Formatting.Value
   --      ("2018-05-01 15:00:00.00", TZ);

end Del.Operators;
