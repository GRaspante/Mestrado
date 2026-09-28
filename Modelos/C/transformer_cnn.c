#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pesos_t_input.h"
#include "pesos_t_cnn.h"
#include "pesos_t_transformer.h"
#include "pesos_t_lstm.h"
//#include "transformer_cnn.h"

#define IN_CHANNELS 4
#define FILTERS 128
#define FILTERS_DEC_2 64
#define SAMPLE_LENGTH 200
#define SAMPLE_REDUCED 100
#define KERNEL_SIZE 3
#define PAD 1
#define POOL_SIZE 2

#define D_MODEL 128
#define DIM_FF 256
#define SEQ_LEN 100

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f)
#define SIGMOID(x) (1.0f / (1.0f + expf(-x)))
#define MAX(x, y) (((x) >= (y)) ? (x) : (y))
#define OFFSET_UP(x, y) (((x) < (y)) ? 1 : 0)
#define IDX_UP(x) (((x) < 0) ? 0 : (x)) 
 
float sampleAdjusted[IN_CHANNELS][SAMPLE_LENGTH + (KERNEL_SIZE - 1)] = {0};
float enc1Output[FILTERS][SAMPLE_LENGTH] = {0};

float maxpoolOutput[FILTERS][SAMPLE_REDUCED] = {0};

float maxpoolAdjusted[FILTERS][SAMPLE_REDUCED + (KERNEL_SIZE - 1)] = {0};
float enc2Output[FILTERS][SAMPLE_REDUCED] = {0};

float transformer_adjusted[SEQ_LEN][D_MODEL] = {0};
float Q[SEQ_LEN][D_MODEL] = {0};
float K[SEQ_LEN][D_MODEL] = {0};
float V[SEQ_LEN][D_MODEL] = {0};
float attn_scores[SEQ_LEN][SEQ_LEN] = {0};
float attn_out[SEQ_LEN][D_MODEL] = {0};
float norm1_out[SEQ_LEN][D_MODEL] = {0};
float ff1_out[SEQ_LEN][DIM_FF] = {0};
float transf_output[SEQ_LEN][D_MODEL] = {0}; 

float lstmOutput[FILTERS][SAMPLE_REDUCED] = {0};
float lstm_h[FILTERS] = {0}; 
float lstm_c[FILTERS] = {0}; 

float lstmAdjusted[FILTERS][SAMPLE_REDUCED + (KERNEL_SIZE - 1)] = {0};
float channels_first_buffer[D_MODEL][SEQ_LEN] = {0};

float dec1Output[FILTERS][SAMPLE_REDUCED] = {0};

float upsampleOutput[FILTERS][SAMPLE_LENGTH] = {0};

float upsampleAdjusted[FILTERS][SAMPLE_LENGTH + (KERNEL_SIZE - 1)] = {0};
float dec2Output[FILTERS_DEC_2][SAMPLE_LENGTH] = {0};

float r_peaks[SAMPLE_LENGTH] = {0};

void adjust_sample() {
    for (int ch = 0; ch < IN_CHANNELS; ch++){	
        for (int i = 0; i < SAMPLE_LENGTH; i++){		
            sampleAdjusted[ch][i + PAD] = sample[ch][i];
        }
    } 
}

void adjust_enc2_input() {
    for (int ch = 0; ch < FILTERS; ch++){	
        for (int i = 0; i < SAMPLE_REDUCED; i++){		
            maxpoolAdjusted[ch][i + PAD] = maxpoolOutput[ch][i];            
        }
	}
}

void adjust_transformer_input() {
    for (int ch = 0; ch < D_MODEL; ch++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            transformer_adjusted[i][ch] = enc2Output[ch][i]; 
        }
    }
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

void transformer_layer() {
    float scale_factor = 1.0f / sqrtf((float)D_MODEL); 


    for (int i = 0; i < SEQ_LEN; i++) { //Matrizes Q, K e V
        for (int d = 0; d < D_MODEL; d++) {
            float zq = w_q_bias[d], zk = w_k_bias[d], zv = w_v_bias[d];
            for (int c = 0; c < D_MODEL; c++) {
                zq += transformer_adjusted[i][c] * w_q_weight[d][c];
                zk += transformer_adjusted[i][c] * w_k_weight[d][c];
                zv += transformer_adjusted[i][c] * w_v_weight[d][c];
            }
            Q[i][d] = zq; 
            K[i][d] = zk; 
            V[i][d] = zv;
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) { // softmax(Q*K')/sqrt(d)
        float max_score = -1e20f;
        for (int j = 0; j < SEQ_LEN; j++) {
            float soma = 0.0f;
            for (int d = 0; d < D_MODEL; d++) {
                soma += Q[i][d] * K[j][d];
            }
            attn_scores[i][j] = soma * scale_factor;
            max_score = MAX(max_score, attn_scores[i][j]);       
        }
        
        // Aplica o Softmax na linha
        float soma_exp = 0.0f;
        for (int j = 0; j < SEQ_LEN; j++) {
            attn_scores[i][j] = expf(attn_scores[i][j] - max_score);
            soma_exp += attn_scores[i][j];
        }
        for (int j = 0; j < SEQ_LEN; j++) {
            attn_scores[i][j] /= soma_exp;
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) { //V * (softmax(Q*K'))
        float temp_attn[D_MODEL] = {0}; 
        
        for (int d = 0; d < D_MODEL; d++) {
            float soma = 0.0f;
            for (int j = 0; j < SEQ_LEN; j++) {
                soma += attn_scores[i][j] * V[j][d];
            }
            temp_attn[d] = soma;
        }
        
        for (int d = 0; d < D_MODEL; d++) { //Projeção
            float proj_soma = out_proj_bias[d];
            for (int k = 0; k < D_MODEL; k++) {
                proj_soma += temp_attn[k] * out_proj_weight[d][k];
            }
            attn_out[i][d] = proj_soma;
        }
    }

    
    for (int i = 0; i < SEQ_LEN; i++) { //Normalização
        float mean = 0.0f, var = 0.0f;

        for (int d = 0; d < D_MODEL; d++) {
            norm1_out[i][d] = transformer_adjusted[i][d] + attn_out[i][d]; 
            mean += norm1_out[i][d];
        }
        mean /= D_MODEL;

        for (int d = 0; d < D_MODEL; d++) {
            var += (norm1_out[i][d] - mean) * (norm1_out[i][d] - mean);
        }
        var /= D_MODEL;
        
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D_MODEL; d++) {
            norm1_out[i][d] = norm1_weight[d] * ((norm1_out[i][d] - mean) * inv_std) + norm1_bias[d];
        }        
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int f = 0; f < DIM_FF; f++) { //FF
            float soma = ff1_bias[f];
            for (int d = 0; d < D_MODEL; d++) {
                soma += norm1_out[i][d] * ff1_weight[f][d];
            }
            ff1_out[i][f] = RELU(soma);
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) { //FF
        for (int d = 0; d < D_MODEL; d++) {
            float soma = ff2_bias[d];
            for (int f = 0; f < DIM_FF; f++) {
                soma += ff1_out[i][f] * ff2_weight[d][f];
            }
            transf_output[i][d] = soma;
        }
    }
    
    for (int i = 0; i < SEQ_LEN; i++) { //Normalização
        float mean = 0.0f, var = 0.0f;

        for (int d = 0; d < D_MODEL; d++) {
            transf_output[i][d] += norm1_out[i][d]; 
            mean += transf_output[i][d];
        }
        mean /= D_MODEL;

        for (int d = 0; d < D_MODEL; d++) {
            var += (transf_output[i][d] - mean) * (transf_output[i][d] - mean);
        }
        var /= D_MODEL;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D_MODEL; d++) {
            transf_output[i][d] = norm2_weight[d] * ((transf_output[i][d] - mean) * inv_std) + norm2_bias[d];
        }        
    }
}

void lstm_layer() {
    float gates[512];

    for (int t = 0; t < SEQ_LEN; t++) {        
  
        for (int j = 0; j < 4 * D_MODEL; j++) {
            float soma = lstm_b_ih[j] + lstm_b_hh[j]; 
            
            for (int k = 0; k < D_MODEL; k++) {
                soma += lstm_w_ih[j * D_MODEL + k] * transf_output[t][k];
                soma += lstm_w_hh[j * D_MODEL + k] * lstm_h[k];
            }
            gates[j] = soma;
        }

        for (int j = 0; j < D_MODEL; j++) {
            float i_gate = SIGMOID(gates[0 * D_MODEL + j]);
            float f_gate = SIGMOID(gates[1 * D_MODEL + j]);
            float g_gate = tanhf(gates[2 * D_MODEL + j]);
            float o_gate = SIGMOID(gates[3 * D_MODEL + j]);

            lstm_c[j] = f_gate * lstm_c[j] + i_gate * g_gate;
            lstm_h[j] = o_gate * tanhf(lstm_c[j]);            

            lstmAdjusted[j][t + PAD] = lstm_h[j];
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
    float scale = 0.5f;
    float offset_h = SAMPLE_REDUCED - 1;
    for (int ch = 0; ch < FILTERS; ch++) {
        for (int i = 0; i < SAMPLE_LENGTH; i++) {
            float pos_idx = (i + 0.5f) * scale - 0.5f;
            pos_idx = IDX_UP(pos_idx);
			
			int offset = OFFSET_UP(pos_idx, offset_h);
            int pos_idx_0 = (int)pos_idx;
            int pos_idx_1 = pos_idx_0 + offset;

            float lambda_1 = pos_idx - (float)pos_idx_0;
            float lambda_0 = 1.0f - lambda_1;

            float val_0 = dec1Output[ch][pos_idx_0];
            float val_1 = dec1Output[ch][pos_idx_1];

            upsampleOutput[ch][i] = (val_0 * lambda_0) + (val_1 * lambda_1);
        }
    }
}

void decoder_conv2() {
    for (int out_ch = 0; out_ch < FILTERS_DEC_2; out_ch++) {
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
        for (int in_ch = 0; in_ch < FILTERS_DEC_2; in_ch++) {
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
    
    adjust_transformer_input();
    transformer_layer();
    lstm_layer();  
    
    decoder_conv1();
    upsample1d();
    
    adjust_dec2_input();
    decoder_conv2();
	r_head();

    return 0;
}
