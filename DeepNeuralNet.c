#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    int input_dim;
    double *w;
    double b;
    double *dw;
    double db;
    double output;
} Neuron;

typedef struct {
    Neuron *neurons;
    int num_inputs;
    int num_outputs;
    double *inputs_cache;
} Layers;

typedef struct {
    Layers *layers;
    int num_layers;
} MLP;

typedef struct {
    double pixels[784];
    int label;
} MNIST_Image;

void init_neuron(Neuron *n, int input_dim) {
    n->input_dim = input_dim;
    n->w = (double *)malloc(input_dim * sizeof(double));
    n->dw = (double *)malloc(input_dim * sizeof(double));
    
    for (int i = 0; i < input_dim; i++) {
        n->w[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        n->dw[i] = 0.0;
    }
    
    n->b = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    n->db = 0.0;
    n->output = 0.0;
}

double call_neuron(Neuron *n, double *in) {
    double sum = 0;
    for (int i = 0; i < n->input_dim; i++) {
        sum += n->w[i] * in[i];
    }
    return sum + n->b;
}

double tanh_activation(double x) {
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

void free_neuron(Neuron *n) {
    free(n->w);
    free(n->dw);
}

void init_layer(Layers *l, int num_inputs, int num_outputs) {
    l->num_inputs = num_inputs;
    l->num_outputs = num_outputs;
    l->inputs_cache = NULL;
    l->neurons = (Neuron *)malloc(num_outputs * sizeof(Neuron));

    for (int i = 0; i < num_outputs; i++) {
        init_neuron(&l->neurons[i], num_inputs);
    }
}

double *call_layer(Layers *l, double *in) {
    if (l->inputs_cache != NULL) free(l->inputs_cache);
    
    l->inputs_cache = (double *)malloc(l->num_inputs * sizeof(double));
    for (int i = 0; i < l->num_inputs; i++) {
        l->inputs_cache[i] = in[i];
    }

    double *out = (double *)malloc(l->num_outputs * sizeof(double));
    for (int i = 0; i < l->num_outputs; i++) {
        double raw_z = call_neuron(&l->neurons[i], in);
        l->neurons[i].output = tanh_activation(raw_z);
        out[i] = l->neurons[i].output;
    }
    
    return out;
}

void free_layer(Layers *l) {
    for (int i = 0; i < l->num_outputs; i++) {
        free_neuron(&l->neurons[i]);
    }
    free(l->neurons);
    if (l->inputs_cache != NULL) free(l->inputs_cache);
}

void init_mlp(MLP *mlp, int num_layers, int *layer_sizes) {
    mlp->num_layers = num_layers - 1;
    mlp->layers = (Layers *)malloc(mlp->num_layers * sizeof(Layers));

    for (int i = 0; i < mlp->num_layers; i++) {
        init_layer(&mlp->layers[i], layer_sizes[i], layer_sizes[i + 1]);
    }
}

double MSE(double *predictions, double *targets, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += pow(predictions[i] - targets[i], 2);
    }
    return sum / size;
}

double *backward_layer(Layers *l, double *d_outputs) {
    double *d_inputs = (double *)malloc(l->num_inputs * sizeof(double));
    for (int i = 0; i < l->num_inputs; i++) {
        d_inputs[i] = 0.0;
    }

    for (int i = 0; i < l->num_outputs; i++) {
        Neuron *n = &l->neurons[i];
        
        double dtanh = 1.0 - (n->output * n->output);
        double error = d_outputs[i] * dtanh;
        
        for (int j = 0; j < n->input_dim; j++) {
            n->dw[j] += error * l->inputs_cache[j];
            d_inputs[j] += error * n->w[j];
        }
        n->db += error;
    }
    
    return d_inputs;
}

void backward_mlp(MLP *mlp, double *d_final_outputs) {
    double *current_d_outputs = d_final_outputs;
    double *next_d_outputs;

    for (int i = mlp->num_layers - 1; i >= 0; i--) {
        next_d_outputs = backward_layer(&mlp->layers[i], current_d_outputs);
        
        if (i < mlp->num_layers - 1) {
            free(current_d_outputs);
        }
        
        current_d_outputs = next_d_outputs;
    }
    
    free(current_d_outputs);
}

double *call_mlp(MLP *mlp, double *in) {
    double *current_input = in;
    double *current_output;

    for (int i = 0; i < mlp->num_layers; i++) {
        current_output = call_layer(&mlp->layers[i], current_input);
        
        if (i > 0) {
            free(current_input);
        }
        current_input = current_output;
    }
    
    return current_output;
}

void update_mlp(MLP *mlp, double learning_rate) {
    for (int i = 0; i < mlp->num_layers; i++) {
        Layers *l = &mlp->layers[i];
        
        for (int j = 0; j < l->num_outputs; j++) {
            Neuron *n = &l->neurons[j];
            
            for (int k = 0; k < n->input_dim; k++) {
                n->w[k] -= learning_rate * n->dw[k];
                n->dw[k] = 0.0;
            }
            
            n->b -= learning_rate * n->db;
            n->db = 0.0;
        }
    }
}

void free_mlp(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        free_layer(&mlp->layers[i]);
    }
    free(mlp->layers);
}

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

MNIST_Image *load_mnist(const char *image_filename, const char *label_filename, int *num_images) {
    FILE *image_file = fopen(image_filename, "rb");
    FILE *label_file = fopen(label_filename, "rb");

    if (!image_file || !label_file) {
        printf("Error: Could not open MNIST files.\n");
        exit(1);
    }

    int magic_number, num_items, num_rows, num_cols, label_magic, label_items;
    
    fread(&magic_number, sizeof(int), 1, image_file);
    magic_number = reverse_int(magic_number);
    
    fread(&num_items, sizeof(int), 1, image_file);
    num_items = reverse_int(num_items);
    
    fread(&num_rows, sizeof(int), 1, image_file);
    num_rows = reverse_int(num_rows);
    
    fread(&num_cols, sizeof(int), 1, image_file);
    num_cols = reverse_int(num_cols);

    fread(&label_magic, sizeof(int), 1, label_file);
    fread(&label_items, sizeof(int), 1, label_file);
    label_items = reverse_int(label_items);

    MNIST_Image *dataset = (MNIST_Image *)malloc(num_items * sizeof(MNIST_Image));
    *num_images = num_items;

    for (int i = 0; i < num_items; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, label_file);
        dataset[i].label = (int)label;

        for (int p = 0; p < 784; p++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, image_file);
            dataset[i].pixels[p] = (double)pixel / 255.0;
        }
    }

    fclose(image_file);
    fclose(label_file);
    return dataset;
}

void one_hot_encode(int label, double *target_array) {
    for (int i = 0; i < 10; i++) {
        target_array[i] = -1.0;
    }
    target_array[label] = 1.0;
}

int argmax(double *array, int size) {
    int best_index = 0;
    double max_val = array[0];
    
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            best_index = i;
        }
    }
    return best_index;
}

int main() {
    srand(time(NULL));

    printf("Loading MNIST Dataset...\n");
    int num_images;
    MNIST_Image *dataset = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &num_images);
    printf("Successfully loaded %d images.\n\n", num_images);

    int layer_sizes[] = {784, 128, 10};
    MLP mlp;
    init_mlp(&mlp, 3, layer_sizes);

    int epochs = 10;
    int train_size = 5000;
    double learning_rate = 0.01;
    double target_array[10];

    printf("Beginning Training Loop...\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct_predictions = 0;
        double total_loss = 0.0;

        for (int i = 0; i < train_size; i++) {
            one_hot_encode(dataset[i].label, target_array);

            double *pred = call_mlp(&mlp, dataset[i].pixels);
            
            if (argmax(pred, 10) == dataset[i].label) {
                correct_predictions++;
            }

            total_loss += MSE(pred, target_array, 10);
            
            double *d_loss = (double *)malloc(10 * sizeof(double));
            for (int j = 0; j < 10; j++) {
                d_loss[j] = 2.0 * (pred[j] - target_array[j]);
            }
            
            backward_mlp(&mlp, d_loss);
            update_mlp(&mlp, learning_rate);
            
            free(pred);
            free(d_loss);
        }

        double accuracy = ((double)correct_predictions / train_size) * 100.0;
        printf("Epoch %d | Loss: %f | Accuracy: %.2f%%\n", epoch + 1, total_loss / train_size, accuracy);
    }

    printf("\nCleaning up memory...\n");
    free_mlp(&mlp);
    free(dataset);

    return 0;
}