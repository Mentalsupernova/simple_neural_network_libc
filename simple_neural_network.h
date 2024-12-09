#ifndef SIMPLE_NEURAL_NETWORK_H
#define SIMPLE_NEURAL_NETWORK_H
#include "libs/simple_matrix_operations.h"
#include "math.h"


typedef enum  {
    SNN_LINEAR = 0,
    SNN_INPUT = 1,
    SNN_OUTPUT,
}LAYER_TYPE;

typedef enum  {
    SNN_RELU = 0,
    SNN_SIGMOID,
    SNN_TANH,
    SNN_SOFTMAX,
    SNN_ACT_INPUT,
    SNN_SELU,
    SNN_LEAKY_RELU
}ACTIVATION_TYPE;

typedef enum  {
  LOSS_NONE = 0,
    SNN_MSE = 1,
    SNN_CROSS_ENTROPY
}LOSS_FUNCTION_TYPE;

typedef struct simple_neural_network_layer{
    uint8_t layer_type;         
    uint8_t activation_type;    
    simple_matrix weights;     
    simple_matrix bias;        
    simple_matrix lo;        
    simple_matrix dweights;     
    simple_matrix dbias;     

  struct simple_neural_network_layer * next;
  struct simple_neural_network_layer * prev;
} simple_neural_network_layer_t; 


typedef struct {
    size_t num_layers;                          

    simple_neural_network_layer_t *first_layer;      

    uint8_t output_function_type;  
    uint8_t loss_function_type;  
    float learning_rate;         
} simple_neural_network_t;

extern void relu(simple_matrix *mat);
extern void selu(simple_matrix *mat);
extern void selu_derivative(simple_matrix *matrix, simple_matrix *output);
extern void apply_relu(simple_neural_network_layer_t *layer);
extern void relu_derivative(simple_matrix *matrix, simple_matrix *output);
extern void apply_selu(simple_neural_network_layer_t *layer);
extern void softmax(simple_matrix *logits);
extern double cross_entropy_loss(simple_matrix *Y,simple_matrix * preds);
extern double add_exp_to_matrix(double value);
extern void print_neural_network(simple_neural_network_t * nn);
extern simple_neural_network_t  allocate_neural_network(float learning_rate,LOSS_FUNCTION_TYPE);
extern void append_input_layer(simple_neural_network_t *nn,simple_matrix  X);
extern void append_layer(simple_neural_network_t *nn, size_t output_neurons,ACTIVATION_TYPE activation_type,LAYER_TYPE layer_type,uint8_t weights_initializer,uint8_t bias_initializer);
extern void forward(simple_neural_network_t *nn); // returns loss
extern void apply_softmax(simple_neural_network_layer_t *layer);
extern void backward(simple_neural_network_t * nn,simple_matrix * Y);
extern void backward_linear(simple_neural_network_layer_t *layer);
extern void backward_output_cross_entropy(simple_neural_network_layer_t *layer, simple_matrix *Y_true);
extern void update_weights_and_biases(simple_neural_network_t *nn);
extern void clip_gradients(simple_matrix *grad, double threshold);
extern simple_matrix inference(simple_neural_network_t *nn, simple_matrix *input_row);
extern double calculate_accuracy_ohe(simple_matrix *preds, simple_matrix *labels);
extern void calculate_gradient_stats(simple_matrix *grad, double *mean, double *std_dev);

/**
 * purpouse of this library to train neural networks using c
 * it depends on simple_matrix_operations https://github.com/Mentalsupernova/simple_matrix_operations
 * this library is header only u need to include header into your project and
 * define SIMPLE_NEURAL_NETWORK_IMPLEMENTATION
 * also link it with math and pthread
 *
 * to start u call allocate_neural_network
 * then u need to define input layer through append_input_layer then add ur layers
 * forward - for forward propogation
 * backward- for backward propogation
 * before inference call forward last time
 * for loss checking use forward as well
 * 
 *
 */
   

#ifdef SIMPLE_NEURAL_NETWORK_H
void clip_gradients_dynamic(simple_matrix *grad, double scale_factor) {
    double mean, std_dev;
    calculate_gradient_stats(grad, &mean, &std_dev);

    double threshold = mean + scale_factor * std_dev;

    for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(grad); i++) {
        if (grad->mat[i] > threshold) {
            grad->mat[i] = threshold;
        } else if (grad->mat[i] < -threshold) {
            grad->mat[i] = -threshold;
        }
    }
}
void calculate_gradient_stats(simple_matrix *grad, double *mean, double *std_dev) {
    size_t total_elements = MATRIX_TOTAL_ELEMENTS(grad);

    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < total_elements; i++) {
        sum += grad->mat[i];
        sum_sq += grad->mat[i] * grad->mat[i];
    }

    *mean = sum / total_elements;
    double variance = (sum_sq / total_elements) - (*mean * *mean);
    *std_dev = sqrt(fmax(variance, 0.0)); 
}
void apply_selu(simple_neural_network_layer_t *layer) {
    simple_matrix *lo = &layer->lo;
    selu(lo);
}
double calculate_accuracy_ohe(simple_matrix *preds, simple_matrix *labels) {
    if (preds->dims[0] != labels->dims[0] || preds->dims[1] != labels->dims[1]) {
        printf("Error: Dimension mismatch between predictions and labels.\n");
        exit(1);
    }

    size_t correct_count = 0;
    for (size_t i = 0; i < preds->dims[0]; i++) {
        size_t pred_argmax = 0;
        double max_val = preds->mat[i * preds->dims[1]];
        
        for (size_t j = 1; j < preds->dims[1]; j++) {
            if (preds->mat[i * preds->dims[1] + j] > max_val) {
                max_val = preds->mat[i * preds->dims[1] + j];
                pred_argmax = j;
            }
        }

        size_t label_argmax = 0;
        max_val = labels->mat[i * labels->dims[1]];
        for (size_t j = 1; j < labels->dims[1]; j++) {
            if (labels->mat[i * labels->dims[1] + j] > max_val) {
                max_val = labels->mat[i * labels->dims[1] + j];
                label_argmax = j;
            }
        }

        if (pred_argmax == label_argmax) {
            correct_count++;
        }
    }

    // Calculate and return accuracy
    return (double)correct_count / preds->dims[0];
}
void selu_derivative(simple_matrix *matrix, simple_matrix *output) {
    if (!matrix || !matrix->mat || !output || !output->mat) {
        printf("Error: Invalid input or output matrix.\n");
        return;
    }

    const float lambda = 1.0507; // Scaling factor
    const float alpha = 1.67326; // Negative slope

    size_t total_elements = MATRIX_TOTAL_ELEMENTS(matrix);
    for (size_t i = 0; i < total_elements; i++) {
        if (matrix->mat[i] >= 0) {
            output->mat[i] = lambda;
        } else {
            output->mat[i] = lambda * alpha * exp(matrix->mat[i]);
        }
    }
}



double add_exp_to_matrix(double value){
  return value + 1e-10;
}

void apply_softmax(simple_neural_network_layer_t *layer){
  simple_matrix * lo = &layer->lo;
  softmax(lo);
}

void apply_relu(simple_neural_network_layer_t *layer){
  simple_matrix * lo = &layer->lo;
  relu(lo);
}
void compute_loss_gradient(simple_matrix *Y_pred, simple_matrix *Y_true, simple_matrix *d_output) {
    for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(Y_pred); i++) {
        d_output->mat[i] = Y_pred->mat[i] - Y_true->mat[i];
    }
}
simple_matrix inference(simple_neural_network_t *nn, simple_matrix *input_row) {
    if (!nn || !nn->first_layer) {
        printf("Error: Neural network not initialized.\n");
        exit(1);
    }

    if (input_row->dims[0] != 1) {
        printf("Error: Input row should have exactly one row. Received %zu rows.\n", input_row->dims[0]);
        exit(1);
    }

    simple_matrix current_output = copy_matrix(input_row);

    simple_neural_network_layer_t *layer = nn->first_layer->next; // Start from the first hidden layer
    while (layer) {
        simple_matrix next_output = allocate_matrix(2, layer->lo.dims, ZEROS, 0.0, 0.0);

        dot(&current_output, &layer->weights, &next_output);

        for (size_t j = 0; j < layer->bias.dims[1]; j++) {
            next_output.mat[j] += layer->bias.mat[j];
        }

        switch (layer->activation_type) {

	case SNN_SELU:

            selu(&next_output);
	  break;

        case SNN_RELU:
            relu(&next_output);
            break;
        case SNN_SOFTMAX:
            softmax(&next_output);
            break;
        default:
            printf("Error: Unsupported activation type in inference.\n");
            exit(1);
        }

        free_matrix(&current_output);
        current_output = next_output;

        layer = layer->next;
    }

    return current_output; 
}


void backward_linear(simple_neural_network_layer_t *layer) {
    simple_matrix *input = &layer->prev->lo;
    simple_matrix *d_output = &layer->lo; // Gradients from the next layer
    simple_matrix *d_weights = &layer->dweights;
    simple_matrix *d_bias = &layer->dbias;
    simple_matrix *weights = &layer->weights;

    simple_matrix input_cpy = copy_matrix(input);
    simple_matrix d_output_cpy = copy_matrix(d_output); // Gradients from the next layer
    simple_matrix d_weights_cpy = copy_matrix(d_weights);
    simple_matrix weights_cpy = copy_matrix(weights);
    if (layer->activation_type == SNN_RELU) {
        simple_matrix relu_grad = allocate_matrix(input->ndims, input->dims, ZEROS, 0.0, 0.0);
        relu_derivative(input, &relu_grad);
        elementwise_mul(d_output, &relu_grad, d_output); // Modify d_output in place
    }else if(layer->activation_type == SNN_RELU){
        simple_matrix relu_grad = allocate_matrix(input->ndims, input->dims, ZEROS, 0.0, 0.0);
        selu_derivative(input, &relu_grad);
        elementwise_mul(d_output, &relu_grad, d_output); // Modify d_output in place

    }


    /* // Compute d_weights: d_output * input^T */
    transpose(&d_output_cpy);
    transpose(&d_weights_cpy);
    dot(&d_output_cpy, &input_cpy, &d_weights_cpy);

    for (size_t i = 0; i < d_output->dims[1]; i++) {
      d_bias->mat[i] = sum_column(&d_output_cpy, i);
    }


    transpose(&weights_cpy);
    dot(d_output, &weights_cpy, &layer->prev->lo);
    

clip_gradients(&layer->dweights, 1.0);  // Clip weights gradient
clip_gradients(&layer->dbias, 1.0);    // Clip bias gradient

}


void backward_output_cross_entropy(simple_neural_network_layer_t *layer, simple_matrix *Y_true) {
    if (!layer || !Y_true) {
        printf("SNN_ERROR: Invalid layer or missing ground truth.\n");
        exit(1);
    }

    // Compute gradients of the loss with respect to predictions
    for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(&layer->lo); i++) {
        layer->lo.mat[i] -= Y_true->mat[i];
    }

    // Compute gradient of weights
    simple_matrix input_transposed = copy_matrix(&layer->prev->lo);
    transpose(&input_transposed);
    dot(&input_transposed, &layer->lo, &layer->dweights);

    // Compute gradient of biases
    for (size_t i = 0; i < layer->bias.dims[1]; i++) {
        layer->dbias.mat[i] = sum_column(&layer->lo, i);
    }

    clip_gradients(&layer->dweights, 1.0);  // Clip weights gradient
    clip_gradients(&layer->dbias, 1.0);    // Clip bias gradient


    // Propagate gradients to the previous layer
    simple_matrix weights_transposed = copy_matrix(&layer->weights);
    transpose(&weights_transposed);
    dot(&layer->lo, &weights_transposed, &layer->prev->lo);
}


void update_weights_and_biases(simple_neural_network_t *nn) {
    simple_neural_network_layer_t *layer = nn->first_layer->next;

    while (layer) {
        for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(&layer->weights); i++) {
            layer->weights.mat[i] -= nn->learning_rate * layer->dweights.mat[i];
        }

        for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(&layer->bias); i++) {
            layer->bias.mat[i] -= nn->learning_rate * layer->dbias.mat[i];
        }

        layer = layer->next;
    }
}






void backward(simple_neural_network_t *nn, simple_matrix *Y) {
    // Start from the last layer
    simple_neural_network_layer_t *layer = nn->first_layer;
    while (layer->next != NULL) {
        layer = layer->next;
    }
    // Backpropagate through each layer
    while (layer->prev != NULL) {
      if(layer->layer_type!= SNN_INPUT){

	switch(layer->layer_type){
	case SNN_LINEAR:
	  backward_linear(layer);
	  break;
	case SNN_OUTPUT:
	  backward_output_cross_entropy(layer, Y);
	  break;
	default:
	  printf("SNN_ERROR: Layer is not supported\n");
	  exit(1);
	  break;
	}



	layer = layer->prev;
      }else{
	break;
      }
      
    }
}



void forward_linear(simple_neural_network_layer_t *current_layer) {
    if (!current_layer || !current_layer->prev) {
        printf("SNN_ERROR: Invalid layer or missing previous layer.\n");
        exit(1);
    }

    simple_matrix *input = &current_layer->prev->lo;
    simple_matrix *weights = &current_layer->weights;
    simple_matrix *bias = &current_layer->bias;

    // Ensure matrix multiplication dimensions are valid
    if (input->dims[1] != weights->dims[0]) {
        printf("SNN_ERROR: Dimension mismatch in matrix multiplication.\n");
        printf("Input dims: [%zu, %zu], Weights dims: [%zu, %zu]\n",
               input->dims[0], input->dims[1], weights->dims[0], weights->dims[1]);
        exit(1);
    }

    // Allocate output matrix
    size_t output_dims[2] = {input->dims[0], weights->dims[1]};
    current_layer->lo = allocate_matrix(2, output_dims, ZEROS, 0.0, 0.0);

    // Perform matrix multiplication: lo = input * weights
    dot(input, weights, &current_layer->lo);


    // Add biases
    for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
            current_layer->lo.mat[i * output_dims[1] + j] += bias->mat[j];
        }
    }
}


void forward(simple_neural_network_t *nn){
  if(nn->first_layer){
    //TODO REFACTOR ME
    simple_neural_network_layer_t *current_layer_pointer = nn->first_layer;
    while(current_layer_pointer){
      switch(current_layer_pointer->layer_type){

          case SNN_LINEAR:
	    forward_linear(current_layer_pointer);
	    break;
          case SNN_INPUT:
	    break;
          case SNN_OUTPUT:
	    forward_linear(current_layer_pointer);
	      break;
          default:
	    printf("SNN_ERROR: layer type unsupported  \n ");
	      exit(1);

	      break;

      }
	
      if(current_layer_pointer->layer_type != SNN_INPUT ){
		switch(current_layer_pointer->activation_type){

		case SNN_SELU:

		  apply_selu(current_layer_pointer);
		  break;
		case SNN_RELU: 
		  apply_relu(current_layer_pointer);
	        break;

		case SNN_SOFTMAX: 
		  if(current_layer_pointer->layer_type == SNN_OUTPUT){

		  apply_softmax(current_layer_pointer);

		  }
	        break;

		default:
		  printf("SNN_ERROR: Activation type not supported %d  %d\n",current_layer_pointer->activation_type,current_layer_pointer->layer_type);
		  exit(1);
		  break;
		}
		  
	      }
      if(current_layer_pointer->next){

	current_layer_pointer = current_layer_pointer->next;
      }else{
	current_layer_pointer = NULL;
	
      }
    }
  }else{
    printf("INPUT LAYER NOT FOUND\n");
    exit(1);
  }
}

simple_neural_network_t  allocate_neural_network(float learning_rate,LOSS_FUNCTION_TYPE t) {
    simple_neural_network_t nn  = {0};
    nn.num_layers = 0;
    nn.loss_function_type = t;
    nn.learning_rate = learning_rate;
    return nn;
}

void append_layer(simple_neural_network_t *nn, size_t output_neurons,ACTIVATION_TYPE activation_type,LAYER_TYPE layer_type,uint8_t weights_initializer,uint8_t bias_initializer){
  if(!nn->first_layer){
    printf("SNN ERROR: First initialize input layer\n");
    exit(1);
  }

simple_neural_network_layer_t *new_layer = (simple_neural_network_layer_t *)malloc(sizeof(simple_neural_network_layer_t));
    if (!new_layer) {
        printf("SNN ERROR: Memory allocation failed for new layer.\n");
        exit(1);
    }

    simple_neural_network_layer_t *last_layer = nn->first_layer;

    size_t input_dim;
    size_t dims_bias[2] = {1,output_neurons}; 
    size_t dims_lo[2]; 
    size_t dims_weights[2];
    //TODO REFACTOR ME
    if(nn->num_layers != 1){

      while (last_layer->next ) {
        last_layer = last_layer->next;
      }

    input_dim = last_layer->lo.dims[1];
    dims_weights[0]  = input_dim; 
    dims_weights[1]  = output_neurons; 
    dims_lo[0] = last_layer->lo.dims[0];
    dims_lo[1] = output_neurons;

    }else{


    input_dim = last_layer->lo.dims[1];
    dims_weights[0]  = input_dim; 
    dims_weights[1]  = output_neurons; 
    dims_lo[0] = last_layer->lo.dims[0];
    dims_lo[1] = output_neurons;

    }

    new_layer->layer_type = layer_type;
    new_layer->activation_type = activation_type;
    new_layer->next = NULL;
if (activation_type == SNN_RELU || activation_type == SNN_LEAKY_RELU) {
    float std_dev = sqrt(2.0 / dims_weights[0]);
    new_layer->weights = allocate_matrix(2, dims_weights, RANDOM, 0.0, std_dev);
} else if (activation_type == SNN_TANH || activation_type == SNN_SIGMOID) {
    float limit = sqrt(6.0 / (dims_weights[0] + dims_weights[1]));
    new_layer->weights = allocate_matrix(2, dims_weights, RANDOM, -limit, limit);
} else {
    float limit = sqrt(6.0 / (dims_weights[0] + dims_weights[1]));
    new_layer->weights = allocate_matrix(2, dims_weights, RANDOM, -limit, limit);
}



    new_layer->bias = allocate_matrix(2, dims_bias, bias_initializer, 0.0, 0.1);
    new_layer->dbias = copy_matrix(&new_layer->bias);
    new_layer->lo = allocate_matrix(2, dims_lo, ZEROS, 0.0, 0.0);
    new_layer->dweights = allocate_matrix(2, dims_weights, ZEROS, 0.0, 0.0);

    last_layer->next = new_layer;
    new_layer->prev = last_layer;




    nn->num_layers +=1;


  
}


void append_input_layer(simple_neural_network_t *nn, simple_matrix X) {
    simple_neural_network_layer_t *input_layer = (simple_neural_network_layer_t *)malloc(sizeof(simple_neural_network_layer_t));
    if (!input_layer) {
        printf("Error: Memory allocation failed for input layer.\n");
        return;
    }

    input_layer->layer_type = SNN_INPUT;  // Specify input layer type
    input_layer->activation_type = SNN_ACT_INPUT;  // Specify activation type for input layer
    input_layer->lo = X;  
    input_layer->next = NULL;  
    input_layer->prev = NULL;  

    if (nn->num_layers == 0) {
        nn->first_layer = input_layer;
    } else {
        simple_neural_network_layer_t *current = nn->first_layer;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = input_layer;
        input_layer->prev = current;
    }

    nn->num_layers++;

    printf("Input layer appended. Network now has %ld layers.\n", nn->num_layers);
}


void print_layer_rec(simple_neural_network_layer_t *layer,size_t layer_num){



  printf("%ld OUTPUT SHAPE: ",layer_num);print_shape(&layer->lo);
  printf("%ld WEIGHTS SHAPE: ",layer_num);print_shape(&layer->weights);
  printf("%ld BIAS SHAPE: ",layer_num);print_shape(&layer->bias);
  
  if(layer->next){
    simple_neural_network_layer_t *lp = layer->next;
    print_layer_rec(lp,layer_num+1);
  }
}
void print_neural_network(simple_neural_network_t * nn){

  if(nn->first_layer){
    printf("0 INPUT  SHAPE: ");print_shape(&nn->first_layer->lo);
    if(nn->first_layer->next){
      simple_neural_network_layer_t *lp = nn->first_layer->next;
      print_layer_rec(lp, 1);
    }

  }

}
void clip_gradients(simple_matrix *grad, double threshold) {
    for (size_t i = 0; i < MATRIX_TOTAL_ELEMENTS(grad); i++) {
        if (grad->mat[i] > threshold) {
            grad->mat[i] = threshold;
        } else if (grad->mat[i] < -threshold) {
            grad->mat[i] = -threshold;
        }
    }
}

double cross_entropy_loss(simple_matrix *predictions, simple_matrix *labels) {
    size_t rows = MATRIX_ROWS(predictions);
    size_t cols = MATRIX_COLS(predictions);

    double loss = 0.0;
    const double epsilon = 1e-12; // Small constant to prevent log(0)

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double p = predictions->mat[i * cols + j];
            double y = labels->mat[i * cols + j];

            // Clip the predictions
            p = fmax(epsilon, fmin(1.0 - epsilon, p));

            if (y > 0.0) {
                loss -= y * log(p);
            }
        }
    }

    return loss / rows; // Average loss
}



void selu(simple_matrix *matrix) {
    if (!matrix || !matrix->mat) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    const float lambda = 1.0507; // Scaling factor
    const float alpha = 1.67326; // Negative slope

    size_t total_elements = MATRIX_TOTAL_ELEMENTS(matrix);
    for (size_t i = 0; i < total_elements; i++) {
        if (matrix->mat[i] >= 0) {
            matrix->mat[i] *= lambda;
        } else {
            matrix->mat[i] = lambda * alpha * (exp(matrix->mat[i]) - 1);
        }
    }
}


void softmax(simple_matrix *logits) {
    size_t rows = MATRIX_ROWS(logits);
    size_t cols = MATRIX_COLS(logits);

    for (size_t i = 0; i < rows; i++) {
        double max_val = -INFINITY;
        for (size_t j = 0; j < cols; j++) {
            if (logits->mat[i * cols + j] > max_val) {
                max_val = logits->mat[i * cols + j];
            }
        }

        double sum_exp = 0.0;
        for (size_t j = 0; j < cols; j++) {
            logits->mat[i * cols + j] = exp(logits->mat[i * cols + j] - max_val);
            sum_exp += logits->mat[i * cols + j];
        }

        for (size_t j = 0; j < cols; j++) {
            logits->mat[i * cols + j] /= sum_exp;
        }
    }
}





void relu_derivative(simple_matrix *matrix, simple_matrix *output) {
    if (!matrix || !matrix->mat || !output || !output->mat) {
        printf("Error: Invalid input or output matrix.\n");
        return;
    }

    if (matrix->ndims != output->ndims) {
        printf("Error: Input and output matrices must have the same number of dimensions.\n");
        return;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < matrix->ndims; i++) {
        if (matrix->dims[i] != output->dims[i]) {
            printf("Error: Input and output matrices must have the same dimensions.\n");
            return;
        }
        total_elements *= matrix->dims[i];
    }

    for (size_t i = 0; i < total_elements; i++) {
        output->mat[i] = matrix->mat[i] > 0 ? 1 : 0;
    }
}

void relu(simple_matrix *matrix) {
    if (!matrix || !matrix->mat) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < matrix->ndims; i++) {
        total_elements *= matrix->dims[i];
    }

    for (size_t i = 0; i < total_elements; i++) {
        matrix->mat[i] = matrix->mat[i] > 0 ? matrix->mat[i] : 0;
    }
}


#endif

#endif
