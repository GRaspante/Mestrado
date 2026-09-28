#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pesos_bloco_cnn.h"
//#include "refinamento.h"

#define IN_CHANNELS 5
#define FILTERS 64
#define SAMPLE_LENGTH 200
#define SAMPLE_REDUCED 100
#define KERNEL_SIZE 3
#define PAD 1
#define POOL_SIZE 2

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f)
#define SIGMOID(x) (1.0f / (1.0f + expf(-x)))
#define MAX(x, y) (((x) >= (y)) ? (x) : (y))

float sample[IN_CHANNELS][SAMPLE_LENGTH] = {0}; 

float sampleAdjusted[IN_CHANNELS][SAMPLE_LENGTH + (KERNEL_SIZE - 1)] = {0};
float enc1Output[FILTERS][SAMPLE_LENGTH] = {0};

float maxpoolOutput[FILTERS][SAMPLE_REDUCED] = {0};

float maxpoolAdjusted[FILTERS][SAMPLE_REDUCED + (KERNEL_SIZE - 1)] = {0};
float enc2Output[FILTERS][SAMPLE_REDUCED] = {0};

float lstmOutput[FILTERS][SAMPLE_REDUCED] = {0};
float lstm_h[FILTERS] = {0}; 
float lstm_c[FILTERS] = {0}; 

float lstmAdjusted[FILTERS][SAMPLE_REDUCED + (KERNEL_SIZE - 1)] = {0};
float dec1Output[FILTERS][SAMPLE_REDUCED] = {0};

float upsampleOutput[FILTERS][SAMPLE_LENGTH] = {0};

float upsampleAdjusted[FILTERS][SAMPLE_LENGTH + (KERNEL_SIZE - 1)] = {0};
float dec2Output[FILTERS][SAMPLE_LENGTH] = {0};

float r_peaks[SAMPLE_LENGTH] = {0};

void adjust_sample() {
    for (int ch = 0; ch < IN_CHANNELS; ch++)
        for (int i = 0; i < SAMPLE_LENGTH; i++)
            sampleAdjusted[ch][i + PAD] = sample[ch][i];
}

void adjust_enc2_input() {
    for (int ch = 0; ch < FILTERS; ch++)
        for (int i = 0; i < SAMPLE_REDUCED; i++)
            maxpoolAdjusted[ch][i + PAD] = maxpoolOutput[ch][i];
}

void adjust_dec1_input() {
    for (int ch = 0; ch < FILTERS; ch++)
        for (int i = 0; i < SAMPLE_REDUCED; i++)
            lstmAdjusted[ch][i + PAD] = lstmOutput[ch][i];
}

void adjust_dec2_input() {
    for (int ch = 0; ch < FILTERS; ch++)
        for (int i = 0; i < SAMPLE_LENGTH; i++)
            upsampleAdjusted[ch][i + PAD] = upsampleOutput[ch][i];
}

void encoder_conv1() {
    for (int out_ch = 0; out_ch < FILTERS; out_ch++) {
        for (int i = 0; i < SAMPLE_LENGTH; i++) {
            float soma = encoder_conv1_biases[out_ch];
            for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    soma += sampleAdjusted[in_ch][i + k] * encoder_conv1_weights[out_ch][in_ch][k];
                }
            }
            enc1Output[out_ch][i] = RELU(soma);
        }
    }
}

void maxpool1d() {
    for (int ch = 0; ch < FILTERS; ch++) {
        for (int i = 0; i < SAMPLE_LENGTH; i += POOL_SIZE) {
            maxpoolOutput[ch][i / POOL_SIZE] = MAX(enc1Output[ch][i], enc1Output[ch][i + 1]);
        }
    }
}

void encoder_conv2() {
    for (int out_ch = 0; out_ch < FILTERS; out_ch++) {
        for (int i = 0; i < SAMPLE_REDUCED; i++) {
            float soma = encoder_conv2_biases[out_ch];
            for (int in_ch = 0; in_ch < FILTERS; in_ch++) {
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    soma += maxpoolAdjusted[in_ch][i + k] * encoder_conv2_weights[out_ch][in_ch][k];
                }
            }
            enc2Output[out_ch][i] = RELU(soma);
        }
    }
}

void lstm_layer() {
    float gates[256];    

    for (int t = 0; t < SAMPLE_REDUCED; t++) {
        for (int j = 0; j < 4 * FILTERS; j++) {
            float soma = lstm_b_ih[j] + lstm_b_hh[j]; 
            
            for (int k = 0; k < FILTERS; k++) {
                soma += lstm_w_ih[j * FILTERS + k] * enc2Output[k][t];
                soma += lstm_w_hh[j * FILTERS + k] * lstm_h[k];
            }
            gates[j] = soma;
        }
        
        for (int j = 0; j < FILTERS; j++) {
            float i_gate = SIGMOID(gates[0 * FILTERS + j]);
            float f_gate = SIGMOID(gates[1 * FILTERS + j]);
            float g_gate = tanhf(gates[2 * FILTERS + j]);
            float o_gate = SIGMOID(gates[3 * FILTERS + j]);

            lstm_c[j] = f_gate * lstm_c[j] + i_gate * g_gate;
            lstm_h[j] = o_gate * tanhf(lstm_c[j]);            
            
            lstmOutput[j][t] = lstm_h[j];
        }
    }
}

void decoder_conv1() {
    for (int out_ch = 0; out_ch < FILTERS; out_ch++) {
        for (int i = 0; i < SAMPLE_REDUCED; i++) {
            float soma = decoder_conv1_biases[out_ch];
            for (int in_ch = 0; in_ch < FILTERS; in_ch++) {
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    soma += lstmAdjusted[in_ch][i + k] * decoder_conv1_weights[out_ch][in_ch][k];
                }
            }
            dec1Output[out_ch][i] = RELU(soma);
        }
    }
}

void upsample1d() {
    float scale = (float)SAMPLE_REDUCED / (float)SAMPLE_LENGTH;
    
    for (int ch = 0; ch < FILTERS; ch++) {
        for (int i = 0; i < SAMPLE_LENGTH; i++) {
            float real_x = (i + 0.5f) * scale - 0.5f;
            if (real_x < 0.0f) real_x = 0.0f;

            int x_left = (int)real_x;
            int x_right = x_left + 1;

            if (x_right >= SAMPLE_REDUCED) {
                x_right = SAMPLE_REDUCED - 1;
                if (x_left >= SAMPLE_REDUCED) x_left = SAMPLE_REDUCED - 1;
            }

            float weight_right = real_x - (float)x_left;
            float weight_left = 1.0f - weight_right;

            float val_left = dec1Output[ch][x_left];
            float val_right = dec1Output[ch][x_right];

            upsampleOutput[ch][i] = (val_left * weight_left) + (val_right * weight_right);
        }
    }
}

void decoder_conv2() {
    for (int out_ch = 0; out_ch < FILTERS; out_ch++) {
        for (int i = 0; i < SAMPLE_LENGTH; i++) {
            float soma = decoder_conv2_biases[out_ch];
            for (int in_ch = 0; in_ch < FILTERS; in_ch++) {
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    soma += upsampleAdjusted[in_ch][i + k] * decoder_conv2_weights[out_ch][in_ch][k];
                }
            }
            dec2Output[out_ch][i] = RELU(soma);
        }
    }
}

void r_head() {
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
        float soma = qrs_head_biases[0];
        for (int in_ch = 0; in_ch < FILTERS; in_ch++) {
            soma += dec2Output[in_ch][i] * qrs_head_weights[0][in_ch][0];
        }
        r_peaks[i] = SIGMOID(soma);
    }
}

int main() {   
 
    adjust_sample();
    encoder_conv1();
    maxpool1d();

    adjust_enc2_input();
    encoder_conv2();
    lstm_layer();  

    adjust_dec1_input();  
    decoder_conv1();

    upsample1d();

    adjust_dec2_input();
    decoder_conv2();

    r_head();

    return 0;
}
