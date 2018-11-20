#include "math.h"

#define SERIAL_BUFFER_SIZE 2048
#define out_chunks 5000
#define samples_per_chunk 100
#define chunk_to_python 20
#define nbr_out_variables 11

volatile float buffer_in[samples_per_chunk];
volatile float error_buffer_out[out_chunks];
volatile float variables_buffer_out[nbr_out_variables];

volatile int n = 0;
volatile int p = 0;
volatile long k = 0;

// Terminos PID
float termino_p = 0.0;
float termino_i = 0.0;
float termino_d = 0.0;

float setpoint = 2.3;
float kp = 1.0;
float ki = 5.0;
float kd = 4.05;
float isteps  = 200;
float isteps_ant = 0;
float dt = 0.0;

///////////////////////////
///////////////////////////

int clock_pin_in = 7;
int chunk_pin_in = 6;
int send_to_python_pin_in = 5;
int recive_from_python_pin_in = 4;

int clock_pin_out = 35;
int chunk_pin_out = 37;
int send_to_python_pin_out = 39;
int analog_read_pin = 0;

float clock_frequency = 160000;
int clock_ind = 0;
float chunk_frequency = 0.;
long chunk_ind = 0;
float send_to_python_frequency = 0.;
long send_to_python_ind = 0;

int m = 0;
int q = 0;
float dummy = 0.;
int vali = 0;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(2*9600);
  //Serial.setTimeout(100);

  analogReadResolution(12);
  analogWriteResolution(12);
  pinMode (DAC0,OUTPUT);
  pinMode (DAC1,OUTPUT);
  analogWrite(DAC1, 2010);  

  
  pinMode (clock_pin_in, INPUT);
  pinMode (chunk_pin_in, INPUT);
  pinMode (send_to_python_pin_in, INPUT);

  pinMode(clock_pin_out, OUTPUT);
  pinMode(chunk_pin_out, OUTPUT);
  pinMode(send_to_python_pin_out, OUTPUT);
  
  // Setea las frecuencias de los clocks
  int divi = 100;
  clock_ind = round(84000000/divi/clock_frequency);
  clock_frequency = round(84000000/divi)/clock_ind;
  chunk_frequency = clock_frequency/samples_per_chunk;
  chunk_ind = round(84000000/divi/chunk_frequency);
  chunk_frequency = round(84000000/divi)/chunk_ind;
  send_to_python_frequency = clock_frequency/samples_per_chunk/chunk_to_python;
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


  // Seteo los clocks
  REG_PIOC_PDR = 0x3FC;  //B1111111100, PIO Disable Register
  REG_PIOC_ABSR = REG_PIOC_ABSR | 0x3FCu; //B1111111100, Peripheral AB Select Register
  REG_PMC_PCER1 = REG_PMC_PCER1 | 16;
  REG_PWM_ENA = REG_PWM_SR | B1111;
  REG_PWM_CLK = PWM_CLK_PREA(0) | PWM_CLK_DIVA(divi);    // Set the PWM clock rate to 2MHz (84MHz/42). Adjust DIVA for the resolution you are looking for                                                     

  REG_PWM_CMR2 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN39
  REG_PWM_CPRD2 = send_to_python_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY2 = round(send_to_python_ind*0.5); 

  REG_PWM_CMR1 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN37
  REG_PWM_CPRD1 = chunk_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY1 = round(chunk_ind*0.5); 

  REG_PWM_CMR0 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN35
  REG_PWM_CPRD0 = clock_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY0 = round(clock_ind*0.5); 
  
  // Inicializo buffer de medicion y de error
  for (k=0;k<samples_per_chunk;k++){
      buffer_in[k] = 0.0;
  }  

  for (m=0;m<out_chunks;m++){
      error_buffer_out[m] = 0.0;
  }      

  k = 0;
  q = 0;
  n = 0;  
  m = 0;
  p = 0;

  // Mando los callbacks
  attachInterrupt(digitalPinToInterrupt(clock_pin_in), acq_callback, RISING); 
  attachInterrupt(digitalPinToInterrupt(chunk_pin_in), chunk_callback, RISING); 
  attachInterrupt(digitalPinToInterrupt(send_to_python_pin_in), send_to_python_callback, FALLING); 

}

void loop() {
  // put your main code here, to run repeatedly:

    // Recibe de python
    if(Serial.available() > 50){ 
        Serial.readBytes((char*)&setpoint, sizeof(setpoint));
        Serial.readBytes((char*)&kp, sizeof(kp));
        Serial.readBytes((char*)&ki, sizeof(ki)); 
        Serial.readBytes((char*)&kd, sizeof(kd));
        Serial.readBytes((char*)&isteps, sizeof(isteps));  
      }  

}

void acq_callback(){

  float ana = analogRead(analog_read_pin);
  buffer_in[k] = ana; 

  k = k + 1;
  k = k%samples_per_chunk;

}

void chunk_callback(){

    int i = 0;
    int w = 0;

    // Promedio de la medicion
    long suma = 0;
    for (i=0;i<samples_per_chunk;i++){
        suma += buffer_in[i];   
    }
    float avg = suma/samples_per_chunk;

    // Error
    error_buffer_out[n] = setpoint - avg;
  
    // Buffer circular 
    int jj = 0;
    int j  = 0;
    jj = (n-int(isteps) + out_chunks)%out_chunks;
    j = (n-1 + out_chunks)%out_chunks;
  
    termino_p = error_buffer_out[n];
    termino_i += (error_buffer_out[n] - error_buffer_out[jj])*dt;
    termino_d = (error_buffer_out[n] - error_buffer_out[j])/dt;

    // Calcula el termino i
    if (isteps_ant != isteps){
      termino_i = 0.;
      if (jj > n){
        for (w=jj;w<out_chunks;w++){
          termino_i += error_buffer_out[w];
          }
        for (w=0;w<n;w++){
          termino_i += error_buffer_out[w];
          }               
        }else{
          for (w=jj;w<n;w++){
            termino_i += error_buffer_out[w];
            }                 
          }
          termino_i = termino_i*dt;
      }
    
    float control = kp*termino_p + ki*termino_i + kd*termino_d;

    // Guarda el vector de salida
    variables_buffer_out[0] = setpoint;
    variables_buffer_out[1] = kp;
    variables_buffer_out[2] = ki;
    variables_buffer_out[3] = kd;
    variables_buffer_out[4] = isteps;
    variables_buffer_out[5] = termino_p;
    variables_buffer_out[6] = termino_i;
    variables_buffer_out[7] = termino_d;
    variables_buffer_out[8] = avg;
    variables_buffer_out[9] = control;    
    variables_buffer_out[10] = float(n);     

    n = n + 1;
    n = n%out_chunks;

    isteps_ant = isteps;

}

void send_to_python_callback(){

    Serial.write( (byte*)&variables_buffer_out, sizeof(variables_buffer_out) );

}



  



