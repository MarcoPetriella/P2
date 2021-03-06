#include "math.h"

#define SERIAL_BUFFER_SIZE 2048
#define in_chunks 5
#define out_chunks 10000
#define samples_per_chunk 100
#define chunk_send 200
#define nbr_out_variables 10
#define out_variables_chunks 5

long total_samples = in_chunks*samples_per_chunk;
float buffer_in[in_chunks*samples_per_chunk];

float error_buffer_out[out_chunks];
float mean_buffer_out[out_chunks];
float variables_buffer_out[nbr_out_variables][out_variables_chunks];

int clock_pin_in = 4;
int chunk_pin_in = 5;
int send_to_python_pin_in = 6;

int clock_pin_out = 37;
int chunk_pin_out = 39;
int send_to_python_pin_out = 41;

int analog_read_pin = 0;

float clock_frequency = 400000;
int clock_ind = 0;
float chunk_frequency = 0.;
long chunk_ind = 0;
float send_to_python_frequency = 0.;
long send_to_python_ind = 0;

int m = 0;
int n = 0;
int q = 0;
int p = 0;
long k = 0;

// Terminos PID
float termino_p = 0.0;
float termino_i = 0.0;
float termino_d = 0.0;

float setpoint = 2.3;
float kp = 1.0;
float ki = 5.0;
float kd = 4.05;
int isteps  = 200;

float dt = 0.0;

char to_python_serial[400];



float vali = 0;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  analogReadResolution(12);
  analogWriteResolution(12);
  pinMode (DAC0,OUTPUT);
  pinMode (DAC1,OUTPUT);
  analogWrite(DAC1, 3000);  

  
  pinMode (clock_pin_in, INPUT);
  pinMode (chunk_pin_in, INPUT);
  pinMode (send_to_python_pin_in, INPUT);

  pinMode(clock_pin_out, OUTPUT);
  pinMode(chunk_pin_out, OUTPUT);
  pinMode(send_to_python_pin_out, OUTPUT);
  
  attachInterrupt(digitalPinToInterrupt(clock_pin_in), acq_callback, RISING); 
  attachInterrupt(digitalPinToInterrupt(chunk_pin_in), chunk_callback, FALLING); 
  attachInterrupt(digitalPinToInterrupt(send_to_python_pin_in), send_to_python_callback, FALLING); 

  int divi = 100;
  clock_ind = round(84000000/divi/clock_frequency);
  clock_frequency = round(84000000/divi)/clock_ind;
  chunk_frequency = clock_frequency/samples_per_chunk;
  chunk_ind = round(84000000/divi/chunk_frequency);
  chunk_frequency = round(84000000/divi)/chunk_ind;
  send_to_python_frequency = clock_frequency/samples_per_chunk/chunk_send;
  send_to_python_ind = round(84000000/divi/send_to_python_frequency);
  send_to_python_frequency = round(84000000/divi)/send_to_python_ind;

  dt = 1./chunk_frequency;

  Serial.print("clock_frequency: ");
  Serial.println(clock_frequency);
  Serial.print("clock_ind: ");
  Serial.println(clock_ind);

  Serial.print("chunk_frequency: ");
  Serial.println(chunk_frequency);
  Serial.print("chunk_ind: ");
  Serial.println(chunk_ind);

  Serial.print("send_to_python_frequency: ");
  Serial.println(send_to_python_frequency);
  Serial.print("send_to_python_ind: ");
  Serial.println(send_to_python_ind);
  Serial.print("chunk_frequency/send_to_python_frequency: ");
  Serial.println(chunk_frequency/send_to_python_frequency);
  
  Serial.print("buffer_size: ");
  Serial.println(SERIAL_BUFFER_SIZE);

  delay(1000);

  REG_PIOC_PDR = 0x3FC;  //B1111111100, PIO Disable Register
  REG_PIOC_ABSR = REG_PIOC_ABSR | 0x3FCu; //B1111111100, Peripheral AB Select Register
  REG_PMC_PCER1 = REG_PMC_PCER1 | 16;
  REG_PWM_ENA = REG_PWM_SR | B1111;
  REG_PWM_CLK = PWM_CLK_PREA(0) | PWM_CLK_DIVA(divi);    // Set the PWM clock rate to 2MHz (84MHz/42). Adjust DIVA for the resolution you are looking for                                                     
  
  REG_PWM_CMR3 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN41
  REG_PWM_CPRD3 = send_to_python_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY3 = round(send_to_python_ind*0.5); 

  REG_PWM_CMR2 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN39
  REG_PWM_CPRD2 = chunk_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY2 = round(chunk_ind*0.5); 

  REG_PWM_CMR1 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN37
  REG_PWM_CPRD1 = clock_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY1 = round(clock_ind*0.5); 
  

  for (k=0;k<total_samples;k++){
      buffer_in[k] = 0.0;
  }  

  for (m=0;m<out_chunks;m++){
      error_buffer_out[m] = 0.0;
  }      

  for (m=0;m<out_chunks;m++){
      mean_buffer_out[m] = 0.0;
  } 


  //Serial.readBytes((byte*) buffer_acq, sizeof(buffer_acq));

  k = 0;
  q = 0;
  n = 0;  
  m = 0;
  p = 0;

}

void loop() {
  // put your main code here, to run repeatedly:


  

}

void acq_callback(){

  float ana = analogRead(analog_read_pin);
  buffer_in[k] = ana; 

  k = k + 1;
  k = k%total_samples;

}

void chunk_callback(){

    int i = 0;
    long suma = 0;
    for (i=0;i<samples_per_chunk;i++){
        suma += buffer_in[m*samples_per_chunk + i];   
    }
    float avg = suma/samples_per_chunk;

    // Terminos PID
    mean_buffer_out[n] = avg;
    error_buffer_out[n] = setpoint - avg;
  
    // BUffer circular 
    int jj = 0;
    int j  = 0;
    jj = (n-isteps + out_chunks)%out_chunks;
    j = (n-1 + out_chunks)%out_chunks;;
  
    termino_p = error_buffer_out[n];
    termino_i += (error_buffer_out[n] - error_buffer_out[jj])*dt;
    termino_d = (error_buffer_out[n] - error_buffer_out[j])/dt;
    
    float control = kp*termino_p + ki*termino_i + kd*termino_d;

    variables_buffer_out[0][p] = setpoint;
    variables_buffer_out[1][p] = kp;
    variables_buffer_out[2][p] = ki;
    variables_buffer_out[3][p] = kd;
    variables_buffer_out[4][p] = isteps;
    variables_buffer_out[5][p] = termino_p;
    variables_buffer_out[6][p] = termino_i;
    variables_buffer_out[7][p] = termino_d;
    variables_buffer_out[8][p] = avg;
    variables_buffer_out[9][p] = control;    


    vali += 1.;
    analogWrite(DAC0, floor(vali));     

    p = p + 1;
    p = p%out_variables_chunks;

    n = n + 1;
    n = n%out_chunks;

    m = m + 1;
    m = m%in_chunks; 



}

void send_to_python_callback(){


  sprintf(to_python_serial, "%f, %f ,%f, %f, %f, %f, %f, %f, %f, %f \n",variables_buffer_out[0][q], variables_buffer_out[1][q], variables_buffer_out[2][q], variables_buffer_out[3][q], variables_buffer_out[4][q], variables_buffer_out[5][q], variables_buffer_out[6][q], variables_buffer_out[7][q], variables_buffer_out[8][q], variables_buffer_out[9][q]);
  Serial.write(to_python_serial);
  Serial.flush(); 

  //Serial.println(to_python_serial);


  q = q + chunk_frequency/send_to_python_frequency;
  q = q%out_variables_chunks;    

  }
    
  

//
//String escribe_a_serial(String tot, float buffer_out, int out_chunks_v){
//
//  int i = 0; 
//  for (i=0;i<out_chunks;i++){
//      char buffer1[10]=" ";
//      char* formato="%i ,";
//      sprintf(buffer1, formato, i);
//      tot = tot + buffer1;  
//    }
//    char buffer1[10]=" ";
//    char* formato="%i ";
//    sprintf(buffer1, formato, i);    
//    //Serial.println(tot);
//
//    return tot;
//
//  }
  



