with Del;
with Del.Operators;
with Ada.Text_IO; use Ada.Text_IO;
with Orka.Numerics.Singles.Tensors.CPU; use Orka.Numerics.Singles.Tensors.CPU;

procedure sigmoid_test is
  
   package D renames Del;
   package DOp renames Del.Operators;

   procedure TestForward1 (Sig : DOp.Sigmoid_Access_T) is
      Test_Tensor   : D.Tensor_T := To_Tensor ([D.Element_T(-2.0), D.Element_T(-1.0), 0.0, 1.0, 2.0], [1, 5]);
      Manual_Calc   : D.Tensor_T := To_Tensor ([0.119, 0.2689, 0.5, 0.7310, 0.8807], [1, 5]);
   begin
      Put_Line("***********Testing Sigmoid Forward***********");
      Put_Line("Values:");
      Put_Line(Test_Tensor.Image);
      New_Line;
      Put_Line("f(x): ");
      Put_Line(Sig.Forward(Test_Tensor).Image);
      New_Line;
      Put_Line("Wanted: ");
      Put_Line(Manual_Calc.Image);
      New_Line;
   end TestForward1;

   procedure TestBackward1 (Sig : DOp.Sigmoid_Access_T) is
      One_Tensor    : D.Tensor_T := Ones((1, 1));
      Manual_Calc   : D.Tensor_T := To_Tensor([0.1049, 0.1966, 0.25, 0.1966, 0.1049], [1, 5]);
   begin
      Put_Line("***********Testing Sigmoid Backward***********");
      Put_Line("f'(x): ");
      Put_Line(Sig.Backward(One_Tensor).Image);
      New_Line;
      Put_Line("Wanted: ");
      Put_Line(Manual_Calc.Image);
   end TestBackward1;

   Forward_Func : DOp.Sigmoid_Access_T;

begin

   Forward_Func := new DOp.Sigmoid_T;

   TestForward1(Forward_Func);
   TestBackward1(Forward_Func);

end sigmoid_test;