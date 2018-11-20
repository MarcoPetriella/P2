#include "math.h"

#define SERIAL_BUFFER_SIZE 2048
#define out_chunks 5000
#define samples_per_chunk 100
#define chunk_to_python 200
#define chunk_from_python 100
#define nbr_out_variables 10

volatile float buffer_in[samples_per_chunk];
volatile float error_buffer_out[out_chunks];
volatile float variables_buffer_out[nbr_out_variables];

volatile int n = 0;
volatile int p = 0;
volatile long k = 0;

// Terminos PID
volatile float termino_p = 0.0;
volatile float termino_i = 0.0;
volatile float termino_d = 0.0;

float setpoint = 2.3;
float kp = 1.0;
float ki = 5.0;
float kd = 4.05;
float isteps  = 200;

float setpoint_rec = 2.3;
float kp_rec = 1.0;
float ki_rec = 5.0;
float kd_rec = 4.05;
float isteps_rec  = 200;

volatile float dt = 0.0;

///////////////////////////
///////////////////////////

int clock_pin_in = 7;
int chunk_pin_in = 6;
int send_to_python_pin_in = 5;
int recive_from_python_pin_in = 4;

int clock_pin_out = 35;
int chunk_pin_out = 37;
int send_to_python_pin_out = 39;
int recive_from_python_pin_out = 41;
int analog_read_pin = 0;

float clock_frequency = 40000;
int clock_ind = 0;
float chunk_frequency = 0.;
long chunk_ind = 0;
float send_to_python_frequency = 0.;
long send_to_python_ind = 0;
float recive_from_python_frequency = 0.;
long recive_from_python_ind = 0;

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
  analogWrite(DAC1, 3000);  

  
  pinMode (clock_pin_in, INPUT);
  pinMode (chunk_pin_in, INPUT);
  pinMode (send_to_python_pin_in, INPUT);

  pinMode(clock_pin_out, OUTPUT);
  pinMode(chunk_pin_out, OUTPUT);
  pinMode(send_to_python_pin_out, OUTPUT);
  

  int divi = 500;
  clock_ind = round(84000000/divi/clock_frequency);
  clock_frequency = round(84000000/divi)/clock_ind;
  chunk_frequency = clock_frequency/samples_per_chunk;
  chunk_ind = round(84000000/divi/chunk_frequency);
  chunk_frequency = round(84000000/divi)/chunk_ind;
  send_to_python_frequency = clock_frequency/samples_per_chunk/chunk_to_python;
  send_to_python_ind = round(84000000/divi/send_to_python_frequency);
  send_to_python_frequency = round(84000000/divi)/send_to_python_ind;
  recive_from_python_frequency = clock_frequency/samples_per_chunk/chunk_from_python;
  recive_from_python_ind = round(84000000/divi/recive_from_python_frequency);
  recive_from_python_frequency = round(84000000/divi)/recive_from_python_ind;

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

  Serial.print("recive_from_python_frequency: ");
  Serial.println(recive_from_python_frequency);
  Serial.print("recive_from_python_ind: ");
  Serial.println(recive_from_python_ind);

  
  Serial.print("buffer_size: ");
  Serial.println(SERIAL_BUFFER_SIZE);

  delay(1000);

  REG_PIOC_PDR = 0x3FC;  //B1111111100, PIO Disable Register
  REG_PIOC_ABSR = REG_PIOC_ABSR | 0x3FCu; //B1111111100, Peripheral AB Select Register
  REG_PMC_PCER1 = REG_PMC_PCER1 | 16;
  REG_PWM_ENA = REG_PWM_SR | B1111;
  REG_PWM_CLK = PWM_CLK_PREA(0) | PWM_CLK_DIVA(divi);    // Set the PWM clock rate to 2MHz (84MHz/42). Adjust DIVA for the resolution you are looking for                                                     

  REG_PWM_CMR3 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN39
  REG_PWM_CPRD3 = recive_from_python_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY3 = round(recive_from_python_ind*0.5); 
  
  REG_PWM_CMR2 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN39
  REG_PWM_CPRD2 = send_to_python_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY2 = round(send_to_python_ind*0.5); 

  REG_PWM_CMR1 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN37
  REG_PWM_CPRD1 = chunk_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY1 = round(chunk_ind*0.5); 

  REG_PWM_CMR0 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN35
  REG_PWM_CPRD0 = clock_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY0 = round(clock_ind*0.5); 
  

  for (k=0;k<samples_per_chunk;k++){
      buffer_in[k] = 0.0;
  }  

  for (m=0;m<out_chunks;m++){
      error_buffer_out[m] = 0.0;
  }      


  //Serial.readBytes((byte*) buffer_acq, sizeof(buffer_acq));

  k = 0;
  q = 0;
  n = 0;  
  m = 0;
  p = 0;



  attachInterrupt(digitalPinToInterrupt(clock_pin_in), acq_callback, RISING); 
  attachInterrupt(digitalPinToInterrupt(chunk_pin_in), chunk_callback, FALLING); 
  attachInterrupt(digitalPinToInterrupt(send_to_python_pin_in), send_to_python_callback, FALLING); 
  //attachInterrupt(digitalPinToInterrupt(recive_from_python_pin_in), recive_from_python_callback, FALLING); 

}

void loop() {
  // put your main code here, to run repeatedly:

//  Serial.print("hola");
//  delay(10);

    char to_python_serial[200];
    sprintf(to_python_serial, "%f, %f ,%f, %f, %f, %f, %f, %f, %f, %f \n",variables_buffer_out[0], variables_buffer_out[1], variables_buffer_out[2], variables_buffer_out[3], variables_buffer_out[4], variables_buffer_out[5], variables_buffer_out[6], variables_buffer_out[7], variables_buffer_out[8], variables_buffer_out[9]);
    Serial.write(to_python_serial);

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
    long suma = 0;
    for (i=0;i<samples_per_chunk;i++){
        suma += buffer_in[i];   
    }
    float avg = suma/samples_per_chunk;

    // Terminos PID
    error_buffer_out[n] = setpoint - avg;
  
    // BUffer circular 
    int jj = 0;
    int j  = 0;
    jj = (n-int(isteps) + out_chunks)%out_chunks;
    j = (n-1 + out_chunks)%out_chunks;;
  
    termino_p = error_buffer_out[n];
    termino_i += (error_buffer_out[n] - error_buffer_out[jj])*dt;
    termino_d = (error_buffer_out[n] - error_buffer_out[j])/dt;
    
    float control = kp*termino_p + ki*termino_i + kd*termino_d;

    vali += 1;
    analogWrite(DAC1, vali/2); 


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
    

//    Serial.print(avg);
//    if (n==0){
//      Serial.println()
//      }

//    if (not n%out_chunks){
//      Serial.println(q);
//      q = q+1;
//      }

  if (not n){
      //Serial.println(vali);
      vali = 0;
    }

    n = n + 1;
    n = n%out_chunks;


}

void send_to_python_callback(){


    char to_python_serial[200];
    sprintf(to_python_serial, "%f, %f ,%f, %f, %f, %f, %f, %f, %f, %f \n",variables_buffer_out[0], variables_buffer_out[1], variables_buffer_out[2], variables_buffer_out[3], variables_buffer_out[4], variables_buffer_out[5], variables_buffer_out[6], variables_buffer_out[7], variables_buffer_out[8], variables_buffer_out[9]);
    Serial.write(to_python_serial);

    //Serial.write( (byte*)&variables_buffer_out, sizeof(variables_buffer_out) );


  if (not q){
      if(Serial.available() >= 20){ 
          Serial.readBytes((char*)&setpoint, sizeof(setpoint));
          Serial.readBytes((char*)&kp, sizeof(kp));
          Serial.readBytes((char*)&ki, sizeof(ki)); 
          Serial.readBytes((char*)&kd, sizeof(kd));
          Serial.readBytes((char*)&isteps, sizeof(isteps));  
        }

        while (Serial.available()){
          Serial.readBytes((char*)&dummy, sizeof(dummy));
          }
        
//        setpoint = setpoint_rec;
//        kp = kp_rec;
//        ki = ki_rec;
//        kd = kd_rec;
//        isteps = isteps_rec;
  }

  Serial.flush();

 q = q+1;
 q = q%100;

 

}


void recive_from_python_callback(){



}  

  



