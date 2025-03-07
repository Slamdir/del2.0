with Ada.Text_IO;                   use Ada.Text_IO;
with Ada.Exceptions;                use Ada.Exceptions;
with Del.JSON;
with Del.JSON_Export;               use Del.JSON_Export;
with Orka.Numerics.Singles.Tensors;

procedure Save_Trained_Model is
   use Orka.Numerics.Singles.Tensors;
   Sample_Dataset : Dataset_Array(1 .. 2);
begin
   -- Example data
   Sample_Dataset(1).Data := new Tensor_T'(Zeros((2, 3)));
   Sample_Dataset(1).Target := new Tensor_T'(Zeros((1, 1)));

   Sample_Dataset(2).Data := new Tensor_T'(Ones((2, 3)));
   Sample_Dataset(2).Target := new Tensor_T'(Ones((1, 1)));

   -- Export to JSON
   Del.JSON_Export.Export_Dataset_JSON("trained_model.json", Sample_Dataset);
end Save_Trained_Model;
