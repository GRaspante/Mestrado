#ifndef TRANSFORMER_CNN_H
#define TRANSFORMER_CNN_H

#define INPUT_DIM 4
#define SAMPLE_LENGTH 200
#define SAMPLE_REDUCED 100

extern float sample[INPUT_DIM][SAMPLE_LENGTH];
extern float r_peaks[SAMPLE_LENGTH];

void adjust_sample(void);
void encoder_conv1(void);

void maxpool1d(void);

void adjust_enc2_input(void);
void encoder_conv2(void);

void adjust_transformer_input(void);
void transformer_layer(void);

void lstm_layer(void);   

void decoder_conv1(void);
void upsample1d(void);

void adjust_dec2_input(void);
void decoder_conv2(void);

void r_head(void);

static inline void transformer_cnn(void) {  

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
}

#endif
