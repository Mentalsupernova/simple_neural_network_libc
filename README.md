## SIMPLE NEURAL NETWORK LIBRARY FOR Calculate


 ### ABOUT
  purpouse of this library to train neural networks using c
  it depends on simple_matrix_operations https://github.com/Mentalsupernova/simple_matrix_operations
  this library is header only u need to include header into your project and
  define SIMPLE_NEURAL_NETWORK_IMPLEMENTATION
  also link it with math and pthread
 
  to start u call allocate_neural_network
  then u need to define input layer through append_input_layer then add ur layers
  forward - for forward propogation
  backward- for backward propogation
  before inference call forward last time
  for loss checking use forward as well
