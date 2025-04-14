with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Unbounded.Text_IO;
with Ada.Strings.Fixed;
with Ada.Exceptions;

with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Data; use Del.Data;

package body Del.Export is

   procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String; Include_Raw_Predictions : Boolean := False; Include_Grid : Boolean := False) is
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

   -- Export Grid Predictions
   procedure Generate_Grid(Out_S : in out Unbounded_String) is
      X_Min : Float_32 := -1.5;
      X_Max : Float_32 := 1.5;
      Y_Min : Float_32 := -1.5;
      Y_Max : Float_32 := 1.5;
      Step  : Float_32 := 0.02;
      X, Y  : Float_32;
      First_Point : Boolean := True;
   begin
      Out_S := Out_S & To_Unbounded_String("    ""grid"": [") & ASCII.LF;
      X := X_Min;
      while X <= X_Max loop
         Y := Y_Min;
         while Y <= Y_Max loop
            declare
               Input_Batch : Tensor_T := Zeros([1, 2]);
               Max_Value   : Float_32;
               Predicted_Index : Integer := 1;
            begin
               Input_Batch.Set(Tensor_Index'(1,1), X);
               Input_Batch.Set(Tensor_Index'(1,2), Y);

               declare
                  Prediction : constant Tensor_T := Self.Run_Layers(Input_Batch);

            begin
               Max_Value := Prediction.Get((1, 1));
               for J in 2 .. Shape(Prediction)(2) loop
                  if Prediction.Get((1, J)) > Max_Value then
                     Max_Value := Prediction.Get((1, J));
                     Predicted_Index := J;
                  end if;
               end loop;

               if not First_Point then
                  Out_S := Out_S & To_Unbounded_String(",");
               end if;
               Out_S := Out_S & ASCII.LF
                           & To_Unbounded_String("        [")
                           & To_Unbounded_String(Trim(Float_32'Image(X), Left)) & To_Unbounded_String(", ")
                           & To_Unbounded_String(Trim(Float_32'Image(Y), Left)) & To_Unbounded_String(", ")
                           & To_Unbounded_String(Integer'Image(Predicted_Index)) & To_Unbounded_String("]");
               First_Point := False;
            end;
            end;
            Y := Y + Step;
         end loop;
         X := X + Step;
      end loop;
      Out_S := Out_S & ASCII.LF & To_Unbounded_String("    ]");
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
         Generate_Grid(Grid_Export);
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
