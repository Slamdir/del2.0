------------------------------------------------------------------------------
--  export_json.adb
--  Demonstrates building JSON with the older GNATCOLL.JSON interface
--  you posted, and handling Orka CPU_Tensor without direct indexing.
------------------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Streams.Stream_IO;
with GNATCOLL.JSON;
with Orka.Numerics.Singles.Tensors;
with Del;  -- for Del.Tensor_T

procedure Export_Training_JSON is

   ----------------------------------------------------------------------------
   -- 1) Convert a String -> Stream_Element_Array for Ada.Streams.Stream_IO
   ----------------------------------------------------------------------------
   function String_To_SEA (S : String) return Ada.Streams.Stream_Element_Array is
      use Ada.Streams;
      SEA : Stream_Element_Array(1 .. S'Length);
   begin
      for I in S'Range loop
         SEA(I) := Stream_Element(S(I));
      end loop;
      return SEA;
   end String_To_SEA;

   ----------------------------------------------------------------------------
   -- 2) Convert a CPU_Tensor to JSON
   --    If your Orka library provides Flatten(T) or similar, use it here.
   --    Below is a placeholder returning an empty array.
   ----------------------------------------------------------------------------
   function CPU_Tensor_To_JSON (T : Del.Tensor_T) return GNATCOLL.JSON.JSON_Value is
   begin
      -- Return an empty array as a placeholder
      return GNATCOLL.JSON.Create(GNATCOLL.JSON.Empty_Array);
   end CPU_Tensor_To_JSON;

begin
   ----------------------------------------------------------------------------
   -- 3) Build a JSON structure and write it to "training_export.json"
   ----------------------------------------------------------------------------
   declare
      use Ada.Streams.Stream_IO;

      SF : File_Type;

      -- Create an empty root object { }
      Root  : GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
      
      -- Create a "model" object
      Model : GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;

      -- Create a JSON array for "layers"
      Layers : GNATCOLL.JSON.JSON_Value := 
        GNATCOLL.JSON.Create(GNATCOLL.JSON.Empty_Array);

      -- Create a JSON array for "classification_results"
      Classification : GNATCOLL.JSON.JSON_Value :=
        GNATCOLL.JSON.Create(GNATCOLL.JSON.Empty_Array);

      -- A sample Orka Tensor
      Sample_Weights : Del.Tensor_T := Orka.Numerics.Singles.Tensors.Zeros((2,2));

   begin
      -- 3a) Fill the "model" object fields
      GNATCOLL.JSON.Set_Field(Model, "type", GNATCOLL.JSON.Create("Neural Network"));

      -- Example "layers" with one object: { "name": "Input", "neurons": 2 }
      declare
         Layer_1 : GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
      begin
         GNATCOLL.JSON.Set_Field(Layer_1, "name",    GNATCOLL.JSON.Create("Input"));
         GNATCOLL.JSON.Set_Field(Layer_1, "neurons", GNATCOLL.JSON.Create(2));

         -- Append Layer_1 to the Layers array
         GNATCOLL.JSON.Append(Layers, Layer_1);
      end;

      -- Add Layers array to the Model object
      GNATCOLL.JSON.Set_Field(Model, "layers", Layers);

      -- 3b) Trained weights from the tensor
      declare
         Weights_Value : GNATCOLL.JSON.JSON_Value := CPU_Tensor_To_JSON(Sample_Weights);
      begin
         GNATCOLL.JSON.Set_Field(Model, "trained_weights", Weights_Value);
      end;

      -- 3c) Classification results
      declare
         Result_Entry : GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
      begin
         GNATCOLL.JSON.Set_Field(Result_Entry, "input",  GNATCOLL.JSON.Create("0.5, 0.5"));
         GNATCOLL.JSON.Set_Field(Result_Entry, "predicted_output", GNATCOLL.JSON.Create(0.8));
         GNATCOLL.JSON.Append(Classification, Result_Entry);
      end;

      -- 3d) Combine everything in Root
      GNATCOLL.JSON.Set_Field(Root, "model",                  Model);
      GNATCOLL.JSON.Set_Field(Root, "classification_results", Classification);

      -- 3e) Convert final JSON to a String
      declare
         Exported_JSON : String := GNATCOLL.JSON.Write(Root, Compact => True);
      begin
         -- Write to file
         Create(SF, Out_File, "training_export.json");
         Write(SF, String_To_SEA(Exported_JSON));
         Close(SF);

         Ada.Text_IO.Put_Line("Exported JSON to training_export.json");
      end;
   end;
end Export_Training_JSON;
