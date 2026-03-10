# traffic
improving a CNN to recognize traffic signs 


experiment I
first CNN config

 	1 Conv2D  32        
	1 MaxPooling2D 
	Flatten
	1 Hidden layer with 128 neurons
	Otuput layer with softmax

result:
accuracy: 0.9541 - loss: 0.1544 

========================================

experiment II
added more neurons into 1 hidden layer: from 128 to 256:

 	1 Conv2D  32       
	1 MaxPooling2D 
	Flatten
	1 Hidden layer with 256 neurons
	Otuput layer with softmax

result:
accuracy: 0.9723 - loss: 0.0964  
========================================

experiment III
added a second hidden layer with 128 neurons

 	1 Conv2D  32
 	1 MaxPooling2D
 	Flatten
 	1 Hidden layer with 256 neurons
 	1 Hidden layer with 128 neurons
 	Otuput layer with softmax

result:
accuracy: 0.9791 - loss: 0.0952
========================================

experiment IV
added 1 Conv2D + MaxPooling2D block 

 	1 Conv2D  32
 	1 MaxPooling2D
 	1 Conv2D  64
 	1 MaxPooling2D
 	Flatten
 	1 Hidden layer with 256 neurons
 	1 Hidden layer with 128 neurons
 	Otuput layer with softmax

result:
accuracy: 0.9810 - loss: 0.0969
========================================

experiment V
Instead Flatten, tried GlobalAveragePooling

 	1 Conv2D  32
 	1 MaxPooling2D
 	1 Conv2D  64
 	1 MaxPooling2D
 	GlobalAveragePooling
 	1 Hidden layer with 256 neurons
 	1 Hidden layer with 128 neurons
 	Otuput layer with softmax

result:
accuracy: 0.9109 - loss: 0.2722
========================================

experiment VI
last config

 	1 Conv2D  32
 	1 MaxPooling2D
 	1 Conv2D  64
 	1 MaxPooling2D
 	1 Conv2D  128
 	Flatten
 	1 Hidden layer with 256 neurons
 	Otuput layer with softmax


result:
accuracy: 0.9870 - loss: 0.0475
