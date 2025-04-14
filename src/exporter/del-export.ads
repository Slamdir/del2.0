with Del.Model;

package Del.Export is
   procedure Export_To_JSON(Self : in Del.Model.Model; Filename : String; Include_Raw_Predictions : Boolean := False; Include_Grid : Boolean := False);
end Del.Export;
