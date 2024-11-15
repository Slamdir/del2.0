with Ada.Text_IO;
with DL_Linear;
use Ada.Text_IO;
use DL_Linear;

procedure Main is
    Input : Float_Array(1 .. 3) := (1.0, 2.0, 3.0);
    Layer : Linear_Layer;
    Output : Float_Array(1 .. 2);
begin
    -- Initialize the layer with input size 3 and output size 2
    Initialize_Layer(Layer, 3, 2);

    -- Perform forward pass
    Output := Forward(Input, Layer);

    -- Display the output
    for Val of Output loop
        Put_Line(Float'Image(Val));
    end loop;
end Main;
