with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Unbounded.Text_IO;
with Ada.Exceptions;

with Del.Model;
with Del.JSON; use Del.JSON;
with Del.Data; use Del.Data;  

package body Del.Export is

  procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String) is
      use Ada.Text_IO;
      use Ada.Strings.Unbounded;
      use Ada.Strings.Unbounded.Text_IO;

      File : File_Type;
      JSON_Content : Unbounded_String := To_Unbounded_String("{ ""data"": [");
      Data_Export, Labels_Export, Predictions_Export, Outputs_Export : Unbounded_String;
   begin
      Create(File, Out_File, Filename);

      if Del.Model.Get_Dataset(Self) = null then
         Put_Line("Warning: No dataset loaded. Exporting empty JSON.");
         JSON_Content := JSON_Content & To_Unbounded_String("], ""labels"": [], ""predictions"": [], ""outputs"": [] }");
         Put(File, To_String(JSON_Content));
         Close(File);
         return;
      end if;

      declare
         Dataset       : constant Training_Data_Access := Del.Model.Get_Dataset(Self);
         All_Samples   : constant Dataset_Array := Dataset.Get_All_Samples;
      begin
         Data_Export := To_Unbounded_String("");
         Labels_Export := To_Unbounded_String(", ""labels"": [");
         Predictions_Export := To_Unbounded_String(", ""predictions"": [");
         Outputs_Export := To_Unbounded_String(", ""outputs"": [");

         for I in All_Samples'Range loop
            declare
               Sample_Data  : constant Tensor_T := All_Samples(I).Data.all;
               Sample_Label : constant Tensor_T := All_Samples(I).Target.all;
               Prediction   : constant Tensor_T := Self.Run_Layers(Sample_Data);
               Output       : constant Tensor_T := Self.Run_Layers(Prediction);
            begin
               -- Data Export
               Data_Export := Data_Export & To_Unbounded_String("[");
               for J in 1 .. Shape(Sample_Data)(2) loop
                  Data_Export := Data_Export & To_Unbounded_String(Float_32'Image(Sample_Data.Get((1, J))));
                  if J < Shape(Sample_Data)(2) then
                     Data_Export := Data_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Data_Export := Data_Export & To_Unbounded_String("]");

               -- Labels Export (get max index = class label)
               declare
                  Label_Index : Integer := 1;
                  Max_Value   : Float_32 := Sample_Label.Get((1, 1));
               begin
                  for J in 2 .. Shape(Sample_Label)(2) loop
                     if Sample_Label.Get((1, J)) > Max_Value then
                        Max_Value := Sample_Label.Get((1, J));
                        Label_Index := J;
                     end if;
                  end loop;
                  Labels_Export := Labels_Export & To_Unbounded_String(Integer'Image(Label_Index));
               end;

               -- Predictions Export
               Predictions_Export := Predictions_Export & To_Unbounded_String("[");
               for J in 1 .. Shape(Prediction)(2) loop
                  Predictions_Export := Predictions_Export & To_Unbounded_String(Float_32'Image(Prediction.Get((1, J))));
                  if J < Shape(Prediction)(2) then
                     Predictions_Export := Predictions_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Predictions_Export := Predictions_Export & To_Unbounded_String("]");

               -- Outputs Export
               Outputs_Export := Outputs_Export & To_Unbounded_String("[");
               for J in 1 .. Shape(Output)(2) loop
                  Outputs_Export := Outputs_Export & To_Unbounded_String(Float_32'Image(Output.Get((1, J))));
                  if J < Shape(Output)(2) then
                     Outputs_Export := Outputs_Export & To_Unbounded_String(", ");
                  end if;
               end loop;
               Outputs_Export := Outputs_Export & To_Unbounded_String("]");

               -- Add commas between samples
               if I < All_Samples'Last then
                  Data_Export := Data_Export & To_Unbounded_String(",");
                  Labels_Export := Labels_Export & To_Unbounded_String(",");
                  Predictions_Export := Predictions_Export & To_Unbounded_String(",");
                  Outputs_Export := Outputs_Export & To_Unbounded_String(",");
               end if;
            end;
         end loop;
      end;

      JSON_Content := JSON_Content & Data_Export
                     & To_Unbounded_String("]")
                     & Labels_Export
                     & To_Unbounded_String("]")
                     & Predictions_Export
                     & To_Unbounded_String("]")
                     & Outputs_Export
                     & To_Unbounded_String("] }");

      Put(File, To_String(JSON_Content));
      Close(File);
      Put_Line("Model training data and final outputs successfully exported to JSON file: " & Filename);

   exception
      when E : others =>
         Put_Line("Error exporting model data: " & Ada.Exceptions.Exception_Message(E));
   end Export_To_JSON;


end Del.Export;
