with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Unbounded.Text_IO;
with Ada.Strings.Fixed;
with Ada.Exceptions;

with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Data; use Del.Data;

package body Del.Export is

   procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String) is
      use Ada.Text_IO;
      use Ada.Strings.Unbounded;
      use Ada.Strings.Unbounded.Text_IO;
      use Ada.Strings.Fixed;

      File : File_Type;
      JSON_Content : Unbounded_String := To_Unbounded_String("{") & To_Unbounded_String(String'(1 => ASCII.LF)) & To_Unbounded_String("    ""data"": [") & To_Unbounded_String(String'(1 => ASCII.LF));
      Data_Export, Labels_Export : Unbounded_String;

      New_Line_Str : constant Unbounded_String := To_Unbounded_String(String'(1 => ASCII.LF));

      -- Helper to append float formatted nicely
      procedure Append_Float(S : in out Unbounded_String; Value : Float_32) is
         Raw : constant String := Float_32'Image(Value);
      begin
         -- Trim leading spaces from Float_32'Image output
         S := S & To_Unbounded_String(Trim(Raw, Ada.Strings.Left));
      end Append_Float;

   begin
      Create(File, Out_File, Filename);

      if Del.Model.Get_Dataset(Self) = null then
         Put_Line("Warning: No dataset loaded. Exporting empty JSON.");
         JSON_Content := JSON_Content
                         & To_Unbounded_String("    ],") & New_Line_Str
                         & To_Unbounded_String("    ""labels"": []") & New_Line_Str
                         & To_Unbounded_String("}");
         Put(File, To_String(JSON_Content));
         Close(File);
         return;
      end if;

      declare
         Dataset       : constant Training_Data_Access := Del.Model.Get_Dataset(Self);
         All_Samples   : constant Dataset_Array := Dataset.Get_All_Samples;
      begin
         Data_Export := To_Unbounded_String("");
         Labels_Export := To_Unbounded_String("    ""labels"": [") & New_Line_Str & To_Unbounded_String("        ");

         for I in All_Samples'Range loop
            declare
               Sample_Data  : constant Tensor_T := All_Samples(I).Data.all;
               Prediction   : constant Tensor_T := Self.Run_Layers(Sample_Data);
               
               -- Variables to find predicted label
               Predicted_Index : Integer := 1;
               Max_Value       : Float_32 := Prediction.Get((1, 1));
            begin
               -- Data Export
               Data_Export := Data_Export & To_Unbounded_String("        [");
               for J in 1 .. Shape(Sample_Data)(2) loop
                  Append_Float(Data_Export, Sample_Data.Get((1, J)));
                  if J < Shape(Sample_Data)(2) then
                     Data_Export := Data_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Data_Export := Data_Export & To_Unbounded_String("]");

               -- Predicted label (argmax)
               for J in 2 .. Shape(Prediction)(2) loop
                  if Prediction.Get((1, J)) > Max_Value then
                     Max_Value := Prediction.Get((1, J));
                     Predicted_Index := J;
                  end if;
               end loop;
               Labels_Export := Labels_Export & To_Unbounded_String(Integer'Image(Predicted_Index));

               -- Add commas if not last sample
               if I < All_Samples'Last then
                  Data_Export := Data_Export & To_Unbounded_String(",") & New_Line_Str;
                  Labels_Export := Labels_Export & To_Unbounded_String(", ");
               else
                  Data_Export := Data_Export & New_Line_Str;
               end if;
            end;
         end loop;
      end;

      -- Combine all parts
      JSON_Content := JSON_Content & Data_Export
                     & To_Unbounded_String("    ],") & New_Line_Str
                     & Labels_Export & New_Line_Str
                     & To_Unbounded_String("    ]") & New_Line_Str
                     & To_Unbounded_String("}");

      -- Write to file
      Put(File, To_String(JSON_Content));
      Close(File);
      Put_Line("Model training data and predicted labels successfully exported to JSON file: " & Filename);

   exception
      when E : others =>
         Put_Line("Error exporting model data: " & Ada.Exceptions.Exception_Message(E));
   end Export_To_JSON;

end Del.Export;
