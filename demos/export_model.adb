with Del.Model;
with Ada.Text_IO; use Ada.Text_IO;

procedure Export_Model is
   Network : Del.Model.Model; -- Assume the trained model is available
begin
   Put_Line("Exporting trained model...");
   Del.Model.Export_To_JSON(Network, "model_output.json");
   Put_Line("Model successfully exported to JSON file.");
end Export_Model;
