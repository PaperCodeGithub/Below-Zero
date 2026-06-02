#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- HYPERPARAMETERS ---
#define VOCAB_SIZE 10000    // Total number of unique words in our dictionary
#define EMBEDDING_DIM 300   // The size of our continuous vector space (300 coordinates per word)
#define CONTEXT_WINDOW 2    // How many words to look at on the left and right
#define LEARNING_RATE 0.01  // Step size for Gradient Descent

// Forward declaration so the compiler knows this function exists
double _probability_denominator(double *center_word, int dim);

// --- THE WEIGHT MATRICES (Theta) ---
// These two arrays hold the raw floating-point coordinates for our vocabulary.
double U[VOCAB_SIZE][EMBEDDING_DIM]; // Context Word Matrix (Target words)
double V[VOCAB_SIZE][EMBEDDING_DIM]; // Center Word Matrix (Anchor words)

// Initializes the matrices with random noise between -1.0 and 1.0
void init_embeddings() {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            U[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            V[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// Computes the geometric similarity between two vectors.
// A higher positive result means the vectors point in the same direction.
double dot_product(double *vec1, double *vec2, int dim) {
    double result = 0.0;
    for (int i = 0; i < dim; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// --- THE SOFTMAX FUNCTION (Forward Pass) ---
// Calculates P(Context | Center): The probability of the context word given the center word.
double probability(double *center_word, double *vec2, int dim) {
    // 1. Calculate raw similarity (numerator)
    double dot = dot_product(center_word, vec2, dim);
    double numerator = exp(dot); 
    
    // 2. Normalize against the entire dictionary (denominator)
    double denominator = _probability_denominator(center_word, dim);
    
    // 3. Return a strict percentage between 0.0 and 1.0
    double prob = numerator / denominator;
    return prob;
}

// Calculates the global baseline by dotting the center word against EVERY word in the dictionary
double _probability_denominator(double *center_word, int dim) {
    double sum = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        double dot = dot_product(center_word, U[i], dim);
        sum += exp(dot);
    }
    return sum;
}

// --- LOSS CALCULATION ---
// Measures how wrong the network's current predictions are across the entire text sequence
double negative_log_likelihood(int *corpus, int corpus_size) {
    double loss = 0.0;
    int valid_windows = 0;

    // Slide the context window across the text corpus
    for(int i = 0; i < corpus_size; i++){
        int center_idx = corpus[i];
        double sectional_sum = 0.0;

        for(int j = -CONTEXT_WINDOW; j <= CONTEXT_WINDOW; j++){
            if(j == 0) continue; // Skip the center word
            int context_pos = i + j;

            // Ensure the context word is within the text boundaries
            if(context_pos >= 0 && context_pos < corpus_size) {
                int context_idx = corpus[context_pos];
            
                // Get the model's predicted probability for this pair
                double prob = probability(V[center_idx], U[context_idx], EMBEDDING_DIM);
                
                // Add the log of the probability (transforms multiplication to addition)
                sectional_sum += log(prob);
                valid_windows++;
            }
        }
        loss += sectional_sum;
    }
    
    // Return the average NLL. We multiply by -1 to frame it as a minimization problem.
    return - (loss / valid_windows);
}

// --- BACKPROPAGATION ---
// Calculates gradients (Expected vs. Observed) and updates the vector coordinates
void backward_pass(int center_idx, int target_idx){
    
    // Stack-allocated array to accumulate the gradient for the center vector
    double grad_v[EMBEDDING_DIM] = {0.0};

    // Pre-compute the Softmax denominator once to save CPU cycles
    double denom = _probability_denominator(V[center_idx], EMBEDDING_DIM);

    // Loop through the entire vocabulary to calculate Expected vs. Observed errors
    for(int i = 0; i < VOCAB_SIZE; i++){
        
        // 1. EXPECTED: What probability did the model assign to this word?
        double prob = exp(dot_product(V[center_idx], U[i], EMBEDDING_DIM)) / denom;

        // 2. OBSERVED: Did this word actually appear in the text? (1.0 if Yes, 0.0 if No)
        double observed = (i == target_idx) ? 1.0 : 0.0;
        
        // 3. ERROR: The gradient direction
        double error = prob - observed;

        // 4. Update the Context Matrix (U) and accumulate Center gradient (V)
        for(int j = 0; j < EMBEDDING_DIM; j++){
            // Accumulate: error * context_vector
            grad_v[j] += error * U[i][j];
            
            // Gradient Descent for Context: U_new = U_old - (lr * error * V_center)
            U[i][j] -= LEARNING_RATE * error * V[center_idx][j];
        }
    }

    // 5. Update the Center Matrix (V) using the fully accumulated gradient
    for (int d = 0; d < EMBEDDING_DIM; d++) {
        V[center_idx][d] -= LEARNING_RATE * grad_v[d];
    }
}

int main() {
    srand(time(NULL));

    // Initialize the matrices with random noise
    init_embeddings();

    // --- THE MOCK TOKENIZER ---
    // To humans, the sentence is: "the cat sat on the mat"
    // We map each unique word to a strict integer ID.
    // "the" = 0, "cat" = 1, "sat" = 2, "on" = 3, "mat" = 4
    
    int tokenized_corpus[] = {0, 1, 2, 3, 0, 4};
    int corpus_size = sizeof(tokenized_corpus) / sizeof(tokenized_corpus[0]);

    printf("Starting Training...\n\n");

    int epochs = 50;
    
    // --- THE TRAINING LOOP ---
    for (int epoch = 0; epoch < epochs; epoch++) {
        
        // Slide the window across the tokenized text array
        for (int i = 0; i < corpus_size; i++) {
            int center_idx = tokenized_corpus[i];

            for (int j = -CONTEXT_WINDOW; j <= CONTEXT_WINDOW; j++) {
                if (j == 0) continue; // Skip the anchor word
                
                int context_pos = i + j;

                // If the context word is within the bounds of the sentence
                if (context_pos >= 0 && context_pos < corpus_size) {
                    int target_idx = tokenized_corpus[context_pos];
                    
                    // Run backpropagation to adjust the geometry of this specific word pair
                    backward_pass(center_idx, target_idx);
                }
            }
        }

        // Print the loss every 10 epochs to verify Gradient Descent is working
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            double current_loss = negative_log_likelihood(tokenized_corpus, corpus_size);
            printf("Epoch %d | NLL Loss: %f\n", epoch, current_loss);
        }
    }

    printf("\nTraining Complete. Meaning has been vectorized.\n");
    return 0;
}