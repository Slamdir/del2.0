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


   -- Export Grid Predictions
   procedure Generate_Grid(Self : in Del.Model.Model; Out_S : in out Unbounded_String) is
      Dataset     : constant Training_Data_Access := Del.Model.Get_Dataset(Self);
      Data_Tensor : constant Tensor_T := Dataset.Get_Data;

      -- 1) Compute bounds and counts once
      X_Min    : Float_32 := Get_Min(Data_Tensor, 1);
      X_Max    : Float_32 := Get_Max(Data_Tensor, 1);
      Y_Min    : Float_32 := Get_Min(Data_Tensor, 2);
      Y_Max    : Float_32 := Get_Max(Data_Tensor, 2);
      Step     : constant Float_32 := 0.01;

      Num_Y    : constant Positive := Natural((Y_Max - Y_Min) / Step) + 1;
      Num_Classes : constant Positive := Shape(Self.Run_Layers(Zeros([1, 2])))(2);

      -- 2) Precompute the Y‑column
      type Y_Array is array (1 .. Num_Y) of Float_32;
      Y_Values : Y_Array;
      Input_Batch : Tensor_T := Zeros((Num_Y, 2));
      Pred_Counter : array (1 .. Num_Classes) of Natural := (others => 0);

      -- Fill the second column once
      begin
         for J in 1 .. Num_Y loop
            Y_Values(J) := Y_Min + Float_32(J - 1) * Step;
            Input_Batch.Set((J, 2), Y_Values(J));
         end loop;

         Out_S := Out_S & To_Unbounded_String("    ""grid"": [") & ASCII.LF;

         -- 3) Now loop over X, one batch eval per X
         declare
            X : Float_32 := X_Min;
            First_Point : Boolean := True;
         begin
            while X <= X_Max loop
               -- fill column 1 with this X
               for J in 1 .. Num_Y loop
                  Input_Batch.Set((J, 1), X);
               end loop;

               -- one big batch prediction
               declare
                  Batch_Preds : constant Tensor_T := Self.Run_Layers(Input_Batch);  -- shape (Num_Y, Num_Classes)
               begin
                  -- unpack each row
                  -- unpack each row
                  for J in 1 .. Num_Y loop
                     declare
                        Pred_Index : Integer   := 1;
                        Max_Val    : Float_32 := Batch_Preds.Get((J, 1));
                        Y_Value    : Float_32 := Y_Values(J);
                     begin
                        -- find argmax in row J
                        for C in 2 .. Num_Classes loop
                           if Batch_Preds.Get((J, C)) > Max_Val then
                              Max_Val    := Batch_Preds.Get((J, C));
                              Pred_Index := C;
                           end if;
                        end loop;

                        -- track usage
                        Pred_Counter(Pred_Index) := Pred_Counter(Pred_Index) + 1;

                        -- emit JSON entry with that Pred_Index
                        if not First_Point then
                           Out_S := Out_S & To_Unbounded_String(",");
                        end if;
                        Out_S := Out_S
                              & ASCII.LF
                              & To_Unbounded_String("        [")
                              & To_Unbounded_String(Trim(Float_32'Image(X), Left))
                              & To_Unbounded_String(", ")
                              & To_Unbounded_String(Trim(Float_32'Image(Y_Value), Left))
                              & To_Unbounded_String(", ")
                              & To_Unbounded_String(Integer'Image(Pred_Index))
                              & To_Unbounded_String("]");
                        First_Point := False;
                     end;
                  end loop;
               end;

               X := X + Step;
            end loop;
         end;

         Out_S := Out_S & ASCII.LF & To_Unbounded_String("    ]");

         -- 4) Debug: ensure all classes appeared
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
