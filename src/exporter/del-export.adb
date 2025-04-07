with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Unbounded.Text_IO;
with Ada.Strings.Fixed;
with Ada.Exceptions;

with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Data; use Del.Data;

package body Del.Export is

   procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String; Include_Raw_Predictions : Boolean := False) is
      use Ada.Text_IO;
      use Ada.Strings.Unbounded;
      use Ada.Strings.Unbounded.Text_IO;
      use Ada.Strings.Fixed;

      File : File_Type;
      JSON_Content : Unbounded_String := To_Unbounded_String("{") & To_Unbounded_String(String'(1 => ASCII.LF)) & To_Unbounded_String("    ""data"": [") & To_Unbounded_String(String'(1 => ASCII.LF));
      Data_Export, Labels_Export, Predictions_Export : Unbounded_String;

      New_Line_Str : constant Unbounded_String := To_Unbounded_String(String'(1 => ASCII.LF));

      -- Helper to append float formatted nicely
      procedure Append_Float(S : in out Unbounded_String; Value : Float_32) is
         Raw : constant String := Float_32'Image(Value);
      begin
         -- Trim leading spaces
         S := S & To_Unbounded_String(Trim(Raw, Ada.Strings.Left));
      end Append_Float;

   begin
      Create(File, Out_File, Filename);

      if Del.Model.Get_Dataset(Self) = null then
         Put_Line("Warning: No dataset loaded. Exporting empty JSON.");
         JSON_Content := JSON_Content
                         & To_Unbounded_String("    ],") & New_Line_Str
                         & To_Unbounded_String("    ""labels"": []") & New_Line_Str;
         if Include_Raw_Predictions then
            JSON_Content := JSON_Content
                            & To_Unbounded_String(",") & New_Line_Str
                            & To_Unbounded_String("    ""predictions"": []") & New_Line_Str;
         end if;
         JSON_Content := JSON_Content & To_Unbounded_String("}");
         Put(File, To_String(JSON_Content));
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
         Data_Export := To_Unbounded_String("");
         Labels_Export := To_Unbounded_String("    ""labels"": [") & New_Line_Str & To_Unbounded_String("        ");
         if Include_Raw_Predictions then
            Predictions_Export := To_Unbounded_String("    ""predictions"": [") & New_Line_Str;
         end if;

         for I in 1 .. Num_Samples loop
            declare
               Input_Batch : Tensor_T := Zeros((1, Num_Features));
               Prediction : Tensor_T := Zeros((1, Num_Classes));
               Predicted_Index : Integer := 1;
               Max_Value : Float_32;
            begin
               -- Fill the batch tensor
               for J in 1 .. Num_Features loop
                  declare
                     Temp_Index : constant Tensor_Index := Tensor_Index'(1, J);
                     Temp_Value : constant Float_32 := Data_Tensor.Get((I, J));
                  begin
                     Input_Batch.Set(Temp_Index, Temp_Value);
                  end;
               end loop;

               -- Predict
               Prediction := Self.Run_Layers(Input_Batch);
               Max_Value := Prediction.Get((1, 1));

               -- Data Export
               Data_Export := Data_Export & To_Unbounded_String("        [");
               for J in 1 .. Num_Features loop
                  Append_Float(Data_Export, Data_Tensor.Get((I, J)));
                  if J < Num_Features then
                     Data_Export := Data_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Data_Export := Data_Export & To_Unbounded_String("]");

               -- Find Predicted Class
               for J in 2 .. Shape(Prediction)(2) loop
                  if Prediction.Get((1, J)) > Max_Value then
                     Max_Value := Prediction.Get((1, J));
                     Predicted_Index := J;
                  end if;
               end loop;
               Labels_Export := Labels_Export & To_Unbounded_String(Integer'Image(Predicted_Index));

               -- Predictions Export
               if Include_Raw_Predictions then
                  Predictions_Export := Predictions_Export & To_Unbounded_String("        [");
                  for J in 1 .. Shape(Prediction)(2) loop
                     Append_Float(Predictions_Export, Prediction.Get((1, J)));
                     if J < Shape(Prediction)(2) then
                        Predictions_Export := Predictions_Export & To_Unbounded_String(", ");
                     end if;
                  end loop;
                  Predictions_Export := Predictions_Export & To_Unbounded_String("]");
               end if;

               -- Commas
               if I < Num_Samples then
                  Data_Export := Data_Export & To_Unbounded_String(",") & New_Line_Str;
                  Labels_Export := Labels_Export & To_Unbounded_String(", ");
                  if Include_Raw_Predictions then
                     Predictions_Export := Predictions_Export & To_Unbounded_String(",") & New_Line_Str;
                  end if;
               else
                  Data_Export := Data_Export & New_Line_Str;
                  if Include_Raw_Predictions then
                     Predictions_Export := Predictions_Export & New_Line_Str;
                  end if;
               end if;
            end;
         end loop;
      end;

      -- Final JSON combine
      JSON_Content := JSON_Content & Data_Export
                     & To_Unbounded_String("    ],") & New_Line_Str
                     & Labels_Export & New_Line_Str
                     & To_Unbounded_String("    ]");

      if Include_Raw_Predictions then
         JSON_Content := JSON_Content
                         & To_Unbounded_String(",") & New_Line_Str
                         & Predictions_Export & New_Line_Str
                         & To_Unbounded_String("    ]");
      end if;

      JSON_Content := JSON_Content & New_Line_Str & To_Unbounded_String("}");

      -- Write file
      Put(File, To_String(JSON_Content));
      Close(File);

      Put_Line("Model training data and predictions successfully exported to JSON file: " & Filename);

   exception
      when E : others =>
         Put_Line("Error exporting model data: " & Ada.Exceptions.Exception_Message(E));
   end Export_To_JSON;

end Del.Export;
