#include "math.h"

#define SERIAL_BUFFER_SIZE 2048
#define in_chunks 20
#define out_chunks 10000
#define samples_per_chunk 100
#define chunk_save 100
#define chunk_show 200

long total_samples = in_chunks*samples_per_chunk;
float buffer_in[in_chunks*samples_per_chunk];
int in_disponible[in_chunks];

int out_disponible[out_chunks];
float buffer_out[out_chunks];

int clock_pin_in = 4;
int chunk_pin_in = 5;
int process_pin_in = 6;

int clock_pin_out = 37;
int chunk_pin_out = 39;
int process_pin_out = 41;

int analog_read_pin = 0;

float clock_frequency = 400000;
int clock_ind = 0;
float chunk_frequency = 0.;
long chunk_ind = 0;
float process_frequency = 0.;
long process_ind = 0;

int m = 0;
int n = 0;
int q = 0;
long k = 0;
int r = 0;
int j = 0;
int jj = 0;

float termino_p = 0.0;
float termino_i = 0.0;
float termino_d = 0.0;

int isteps  = 200;



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
  pinMode (process_pin_in, INPUT);

  pinMode(clock_pin_out, OUTPUT);
  pinMode(chunk_pin_out, OUTPUT);
  pinMode(process_pin_out, OUTPUT);
  
  attachInterrupt(digitalPinToInterrupt(clock_pin_in), acq_callback, FALLING); 
  attachInterrupt(digitalPinToInterrupt(chunk_pin_in), chunk_callback, FALLING); 
  attachInterrupt(digitalPinToInterrupt(process_pin_in), process_callback, FALLING); 

  int divi = 100;
  clock_ind = round(84000000/divi/clock_frequency);
  clock_frequency = round(84000000/divi)/clock_ind;
  chunk_frequency = clock_frequency/samples_per_chunk;
  chunk_ind = round(84000000/divi/chunk_frequency);
  chunk_frequency = round(84000000/divi)/chunk_ind;
  process_frequency = clock_frequency/samples_per_chunk/chunk_show;
  process_ind = round(84000000/divi/process_frequency);
  process_frequency = round(84000000/divi)/process_ind;

  Serial.print("clock_frequency: ");
  Serial.println(clock_frequency);
  Serial.print("clock_ind: ");
  Serial.println(clock_ind);

  Serial.print("chunk_frequency: ");
  Serial.println(chunk_frequency);
  Serial.print("chunk_ind: ");
  Serial.println(chunk_ind);

  Serial.print("process_frequency: ");
  Serial.println(process_frequency);
  Serial.print("process_ind: ");
  Serial.println(process_ind);
  Serial.print("buffer_size: ");
  Serial.println(SERIAL_BUFFER_SIZE);

  delay(1000);

  REG_PIOC_PDR = 0x3FC;  //B1111111100, PIO Disable Register
  REG_PIOC_ABSR = REG_PIOC_ABSR | 0x3FCu; //B1111111100, Peripheral AB Select Register
  REG_PMC_PCER1 = REG_PMC_PCER1 | 16;
  REG_PWM_ENA = REG_PWM_SR | B1111;
  REG_PWM_CLK = PWM_CLK_PREA(0) | PWM_CLK_DIVA(divi);    // Set the PWM clock rate to 2MHz (84MHz/42). Adjust DIVA for the resolution you are looking for                                                     
  
  REG_PWM_CMR3 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN41
  REG_PWM_CPRD3 = process_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY3 = round(process_ind*0.5); 

  REG_PWM_CMR2 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN39
  REG_PWM_CPRD2 = chunk_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY2 = round(chunk_ind*0.5); 

  REG_PWM_CMR1 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2 - PIN37
  REG_PWM_CPRD1 = clock_ind;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6
  REG_PWM_CDTY1 = round(clock_ind*0.5); 
  

  for (k=0;k<total_samples;k++){
      buffer_in[k] = 0.0;
  }  

  for (q=0;q<in_chunks;q++){
    in_disponible[q] = 0;
    } 

  for (m=0;m<out_chunks;m++){
      buffer_out[m] = 0.0;
  }      

  for (m=0;m<out_chunks;m++){
      out_disponible[m] = 0;
  }   

  //Serial.readBytes((byte*) buffer_acq, sizeof(buffer_acq));

  k = 0;
  q = 0;
  m = 0;
  n = 0;
  r = 0;
  

}

void loop() {
  // put your main code here, to run repeatedly:


  

}

void acq_callback(){

  float ana = analogRead(analog_read_pin);
  buffer_in[k] = ana; 
//  buffer_acq[m][n] = ana;
//  m = k/samples_per_chunk;
//  n = k%samples_per_chunk;
  
  k = k + 1;
  k = k%total_samples;

}

void chunk_callback(){

  in_disponible[m] += 1;

  if (in_disponible[m] > 1){
    Serial.println("Hay Overrun");
    }  

    int i = 0;
    long suma = 0;
    for (i=0;i<samples_per_chunk;i++){
        suma += buffer_in[m*samples_per_chunk + i];   
    }
    float avg = suma/samples_per_chunk;
    buffer_out[n] = avg;
    out_disponible[n] +=  1;
    //Serial.println(avg);
  
    // BUffer circular 
    jj = (n-isteps + out_chunks)%out_chunks;
  
    termino_i += (buffer_out[n] - buffer_out[jj]);
   //Serial.println(buffer_out[n]);

//    if (not n%chunk_save){
//      int i= 0;
//      for (i=0;i<10-1;i++){
//        Serial.print(buffer_out[jj+i]);      
//        }
//        Serial.println(buffer_out[jj+i]);       
////      byte* byteData = (byte*)(&buffer_out[jj]);    // Casting to a byte pointer
////      Serial.write(byteData, 4*chunk_save); 
//    }



  vali += 1.;
  analogWrite(DAC0, floor(vali));     

  
    n = n + 1;
    n = n%out_chunks;

    m = m + 1;
    m = m%in_chunks; 

  in_disponible[m] -= 1;


}

void process_callback(){


  Serial.println(buffer_out[q*chunk_show]);

  
  q = q + 1;
  q = q%(out_chunks/chunk_show);    

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
  



