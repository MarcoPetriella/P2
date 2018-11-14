void setup () { 

   pinMode(39, OUTPUT);
  
  // PWM Set-up on pin PC7 (Arduino Pin 39): see Datasheet chap. 38.5.1
  // Select Instance=PWM; Signal=PWMH2 (channel 2); I/O Line=PC7 (P7, Arduino pine 39, see pinout diagram) ; Peripheral=B

  PMC->PMC_PCER1 |= PMC_PCER1_PID36;                   // PWM on

  REG_PIOC_ABSR |= PIO_ABSR_P7;                        // Set PWM pin perhipheral type B

  REG_PIOC_PDR |= PIO_PDR_P7;                          // Set PWM pin to an output

  REG_PWM_ENA = PWM_ENA_CHID2;                         // Enable the PWM channel 2 (see datasheet page 973) 
 
  REG_PWM_CLK = PWM_CLK_PREA(0) | PWM_CLK_DIVA(1000);    // Set the PWM clock rate to 2MHz (84MHz/42). Adjust DIVA for the resolution you are looking for
                                                      
  REG_PWM_CMR2 = PWM_CMR_CALG |PWM_CMR_CPRE_CLKA;      // The period is left aligned, clock source as CLKA on channel 2

  REG_PWM_CPRD2 = 8000;                             // Channel 2 : Set the PWM frequency 2MHz/(2 * CPRD) = F ; 1< CPRD < 2exp24  -1 ; here CPRD = 10 exp 6

  REG_PWM_CDTY2 = 4000;                              // Channel 2: Set the PWM duty cycle to x%= (CDTY/ CPRD)  * 100 % , CDTY = 2 * 10 exp 5

 // Alternatively, you can use this format :  PWM->PWM_CH_NUM[2].PWM_CPRD = 1000000 ;                    
 // In this example, Frequency is 1 HZ with a DT of 20% so you can see it with an LED attached to pin 39 with a resistor  
 
}


void loop() {
}
