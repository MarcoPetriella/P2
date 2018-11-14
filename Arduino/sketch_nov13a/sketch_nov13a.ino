

void setup() {
  // put your setup code here, to run once:
    Serial.begin(115200);

}

void loop() {
  // put your main code here, to run repeatedly:

  String tot;
  int i = 0;
  for (i=0;i<100;i++){
    char buffer1[10]=" ";
    char* formato="%f ,";
    sprintf(buffer1, formato, float(i));
    tot = tot + buffer1;
    
    }
    Serial.println(tot);
  

}
