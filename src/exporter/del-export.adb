with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Unbounded.Text_IO;
with Ada.Strings.Fixed;
with Ada.Exceptions;
with Del.Export; use Del.Export;
with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Data; use Del.Data;

package body Del.Export is

   procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String; Include_Raw_Predictions : Boolean := False; Include_Grid : Boolean := True) is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Unbounded.Text_IO;
   use Ada.Strings.Fixed;
   use Ada.Strings; -- For Trim

   File : File_Type;
   JSON_Content : Unbounded_String := To_Unbounded_String("{") & To_Unbounded_String(String'(1 => ASCII.LF));
   Data_Export, Labels_Export, Predictions_Export, Grid_Export : Unbounded_String;
   New_Line_Str : constant Unbounded_String := To_Unbounded_String(String'(1 => ASCII.LF));

   -- Helper to append nicely formatted float
   procedure Append_Float(S : in out Unbounded_String; Value : Float_32) is
      Raw : constant String := Float_32'Image(Value);
   begin
      S := S & To_Unbounded_String(Trim(Raw, Ada.Strings.Left));
   end Append_Float;

      function Get_Min(T : Tensor_T; F : Positive) return Float_32 is
      R : Float_32 := T.Get((1, F));
   begin
      for I in 2 .. Shape(T)(1) loop
         if T.Get((I, F)) < R then
            R := T.Get((I, F));
         end if;
      end loop;
      return R;
   end Get_Min;

   function Get_Max(T : Tensor_T; F : Positive) return Float_32 is
      R : Float_32 := T.Get((1, F));
   begin
      for I in 2 .. Shape(T)(1) loop
         if T.Get((I, F)) > R then
            R := T.Get((I, F));
         end if;
      end loop;
      return R;
   end Get_Max;

   procedure Generate_Grid(Self : in Del.Model.Model; Out_S : in out Unbounded_String) is
      -- pull in the dataset
      Dataset     : constant Training_Data_Access := Del.Model.Get_Dataset(Self);
      Data_Tensor : constant Tensor_T := Dataset.Get_Data;

      -- compute real bounds
      X_Min : Float_32 := Get_Min(Data_Tensor, 1);
      X_Max : Float_32 := Get_Max(Data_Tensor, 1);
      Y_Min : Float_32 := Get_Min(Data_Tensor, 2);
      Y_Max : Float_32 := Get_Max(Data_Tensor, 2);

      Step : constant Float_32 := 0.009;

      -- compute integer counts
      Num_X : constant Positive := Positive (Float_32'Floor((X_Max - X_Min) / Step)) + 1;
      Num_Y : constant Positive := Positive (Float_32'Floor((Y_Max - Y_Min) / Step)) + 1;

      -- batch buffer
      Grid_Input : Tensor_T := Zeros((Num_X * Num_Y, 2));
      Num_Classes : constant Positive := Shape(Self.Run_Layers(Zeros((1,2))))(2);
      All_Preds : Tensor_T := Zeros((Num_X * Num_Y, Num_Classes));
      Pred_Counter : array (1 .. Num_Classes) of Natural := (others => 0);

      -- helper to map (i,j) -> linear index
      function Index (I, J : Positive) return Positive is
      begin
         return (I-1) * Num_Y + J;
      end Index;

   begin
      -- fill entire grid in one shot
      for I in 1 .. Num_X loop
         for J in 1 .. Num_Y loop
            Grid_Input.Set((Index(I,J), 1), X_Min + Float_32(I-1)*Step);
            Grid_Input.Set((Index(I,J), 2), Y_Min + Float_32(J-1)*Step);
         end loop;
      end loop;

      -- single batch prediction
      All_Preds := Self.Run_Layers(Grid_Input);

      -- begin JSON
      Out_S := Out_S & To_Unbounded_String("    ""grid"": [") & ASCII.LF;

      -- unpack every row
      declare
         First : Boolean := True;
      begin
         for I in 1 .. Num_X loop
            for J in 1 .. Num_Y loop
               declare
               Pred_Index : Integer   := 1;
               Max_Val    : Float_32 := All_Preds.Get((Index(I, J), 1));
            begin
               -- find argmax in row
               for C in 2 .. Num_Classes loop
                  if All_Preds.Get((Index(I, J), C)) > Max_Val then
                     Max_Val    := All_Preds.Get((Index(I, J), C));
                     Pred_Index := C;
                  end if;
               end loop;

               Pred_Counter(Pred_Index) := Pred_Counter(Pred_Index) + 1;

               -- emit JSON entry
               if not First then
                  Out_S := Out_S & To_Unbounded_String(",");
               end if;
               Out_S := Out_S
                     & ASCII.LF
                     & To_Unbounded_String("        [")
                     & To_Unbounded_String(Trim(Float_32'Image(X_Min + Float_32(I-1)*Step), Left))
                     & To_Unbounded_String(", ")
                     & To_Unbounded_String(Trim(Float_32'Image(Y_Min + Float_32(J-1)*Step), Left))
                     & To_Unbounded_String(", ")
                     & To_Unbounded_String(Integer'Image(Pred_Index))
                     & To_Unbounded_String("]");
               First := False;
            end;
            end loop;
         end loop;
      end;

      Out_S := Out_S & ASCII.LF & To_Unbounded_String("    ]");

      -- print to see if any class is zero
      for K in Pred_Counter'Range loop
         Put_Line("  Grid hit count for class " & Integer'Image(K)
                  & ": " & Natural'Image(Pred_Counter(K)));
      end loop;
   end Generate_Grid;



begin
   Create(File, Out_File, Filename);

   if Del.Model.Get_Dataset(Self) = null then
      Put_Line("Warning: No dataset loaded. Exporting empty JSON.");
      Put_Line(File, "{}");
      Close(File);
      return;
   end if;

   declare
      Dataset       : constant Training_Data_Access := Del.Model.Get_Dataset(Self);
      Data_Tensor   : constant Tensor_T := Dataset.Get_Data;
      Labels_Tensor : constant Tensor_T := Dataset.Get_Labels;
      Num_Samples   : constant Natural := Shape(Data_Tensor)(1);
      Num_Features  : constant Natural := Shape(Data_Tensor)(2);
      Num_Classes   : constant Natural := Shape(Labels_Tensor)(2);
   begin
      -- Export Training Data
      Data_Export := To_Unbounded_String("    ""data"": [") & New_Line_Str;
      Labels_Export := To_Unbounded_String("    ""labels"": [") & New_Line_Str;
      if Include_Raw_Predictions then
         Predictions_Export := To_Unbounded_String("    ""predictions"": [") & New_Line_Str;
      end if;

      for I in 1 .. Num_Samples loop
         declare
            Input_Batch : Tensor_T := Zeros((1, Num_Features));
            Prediction  : Tensor_T := Zeros((1, Num_Classes));
            Predicted_Index : Integer := 1;
            Max_Value : Float_32;
         begin
            for J in 1 .. Num_Features loop
                  declare
                     Temp_Index : constant Tensor_Index := Tensor_Index'(1, J);
                     Temp_Value : constant Float_32 := Data_Tensor.Get((I, J));
                  begin
                     Input_Batch.Set(Temp_Index, Temp_Value);
                  end;
          end loop;

            Prediction := Self.Run_Layers(Input_Batch);
            Max_Value := Prediction.Get((1, 1));
            for J in 2 .. Shape(Prediction)(2) loop
               if Prediction.Get((1, J)) > Max_Value then
                  Max_Value := Prediction.Get((1, J));
                  Predicted_Index := J;
               end if;
            end loop;

            -- Data export
            Data_Export := Data_Export & "        [";
            for J in 1 .. Num_Features loop
               Append_Float(Data_Export, Data_Tensor.Get((I, J)));
               if J < Num_Features then
                  Data_Export := Data_Export & To_Unbounded_String(", ");
               end if;
            end loop;
            Data_Export := Data_Export & To_Unbounded_String("]");

            Labels_Export := Labels_Export & "        " & Integer'Image(Predicted_Index);

            if Include_Raw_Predictions then
               Predictions_Export := Predictions_Export & "        [";
               for J in 1 .. Shape(Prediction)(2) loop
                  Append_Float(Predictions_Export, Prediction.Get((1, J)));
                  if J < Shape(Prediction)(2) then
                     Predictions_Export := Predictions_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Predictions_Export := Predictions_Export & "]";
            end if;

            if I < Num_Samples then
               Data_Export := Data_Export & "," & New_Line_Str;
               Labels_Export := Labels_Export & "," & New_Line_Str;
               if Include_Raw_Predictions then
                  Predictions_Export := Predictions_Export & "," & New_Line_Str;
               end if;
            else
               Data_Export := Data_Export & New_Line_Str;
               Labels_Export := Labels_Export & New_Line_Str;
               if Include_Raw_Predictions then
                  Predictions_Export := Predictions_Export & New_Line_Str;
               end if;
            end if;
         end;
      end loop;

      -- Write sections
      JSON_Content := JSON_Content & Data_Export & "    ]," & New_Line_Str;
      JSON_Content := JSON_Content & Labels_Export & "    ]";

      if Include_Raw_Predictions then
         JSON_Content := JSON_Content & "," & New_Line_Str & Predictions_Export & "    ]";
      end if;

      if Include_Grid then
         JSON_Content := JSON_Content & "," & New_Line_Str;
         Generate_Grid(Self, Grid_Export); 
         JSON_Content := JSON_Content & Grid_Export;
      end if;

      JSON_Content := JSON_Content & New_Line_Str & "}";

      -- Write to file
      Put(File, To_String(JSON_Content));
      Close(File);

      Put_Line("Model and grid exported successfully to " & Filename);
   end;

exception
   when E : others =>
      Put_Line("Error during export: " & Ada.Exceptions.Exception_Message(E));
end Export_To_JSON;

end Del.Export;
