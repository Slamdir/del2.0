with Ada.Real_Time; use Ada.Real_Time;
package body Del.Operators is

   procedure Initialize(L : in out Linear_T; In_Nodes, Out_Nodes : Positive) is
      -- Initialize with uniform random values between -0.1 and 0.1 for stable training
      Weights : Tensor_T := Random_Uniform((In_Nodes, Out_Nodes));
      Bias    : Tensor_T := Random_Uniform((1, Out_Nodes));

      Weights_Grad : Tensor_T := Zeros((In_Nodes, Out_Nodes));
      Bias_Grad : Tensor_T := Zeros((1, Out_Nodes));

      Weights_Velocity : Tensor_T := Zeros((In_Nodes, Out_Nodes));
      Bias_Velocity : Tensor_T := Zeros((1, Out_Nodes));
   begin
      L.Map.Insert("weights", Weights);
      L.Map.Insert("bias", Bias);

      L.Map.Insert("weights_grad", Weights_Grad);
      L.Map.Insert("bias_grad", Bias_Grad);

      L.Map.Insert("weights_velocity", Weights_Velocity);
      L.Map.Insert("bias_velocity", Bias_Velocity);
   end Initialize;

   overriding function Forward (L : in out Linear_T; X : Tensor_T) return Tensor_T is
         Weights : constant Tensor_T := L.Map("weights");
         Bias    : constant Tensor_T := L.Map("bias");

         Temp    : constant Tensor_T := X * Weights;
         Output  : constant Tensor_T := Temp + Bias; 
      begin
         -- Store input for backward pass using Include instead of Insert
         L.Map.Include("input", X);
         return Output;
      end Forward;

   overriding function Backward (L : in out Linear_T; Dy : Tensor_T) return Tensor_T is
      Input      : constant Tensor_T := L.Map("input");
      Weights    : constant Tensor_T := L.Map("weights");
      Batch_Size : constant Positive := Shape(Input)(1);

      -- Get current gradients
      Weights_Grad   : Tensor_T := L.Map("weights_grad");
      Bias_Grad      : Tensor_T := L.Map("bias_grad");

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
      L.Map("weights_grad") := Weights_Grad;
      L.Map("bias_grad")    := Bias_Grad;
      
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

      Exp_Values     : Tensor_T := Exp(X);
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

   --  TZ   : Time_Offset := UTC_Time_Offset;
   --  Zero : Ada.Calendar.Time        :=
   --    Ada.Calendar.Formatting.Value
   --      ("2018-05-01 15:00:00.00", TZ);

   SC : Seconds_Count;
   TS : Time_Span;

begin
   Ada.Real_Time.Split(Clock, SC, TS);
   Reset_Random (To_Duration(TS) * 100000);
end Del.Operators;
