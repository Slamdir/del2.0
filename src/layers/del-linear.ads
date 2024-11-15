-- dl_linear.ads

package Del.Linear is
   -- Define a subtype for array indxs
   subtype Index_Type is Positive;


   -- Linear_Layer type with discriminants to constrain array sizes
   type Linear_Layer is new Layer_T with null record;

   -- Procedure to initialize the layer's weights and biases
   procedure Initialize_Layer (Layer : out Linear_Layer);

   -- Forward pass function
   function Forward (Input : Float_Array; Layer : Linear_Layer) return Float_Array;

   -- Backward pass procedure
   procedure Backward (Input : Float_Array; d_Output : Float_Array; Layer : in out Linear_Layer);

end Del.Linear;
