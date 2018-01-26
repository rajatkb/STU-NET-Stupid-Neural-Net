# STU-NET-Stupid-Neural-Net
STU-NET as the acronym goes is a really bad attempt at making race car A.I . So the attempt is to have the car drift , slide and move with really high speed. Because mainstream self driving a.i is mehh!! 

<h3>
  How i made the model
</h3>
<p>
  <ul>
    <li> Got a lot of hints from Sentdex GTAV A.I tutorial aka CHARLES THE A.I , got scripts like screen capture and keys capture from there</li>
    <li>Next Got some training data by playing the game. The model was quite simple for thic CNN => One hot encoding of movement key classes. So made the gameplay data according to that</li>
    <li>trained on my own model architecture</li>
    <li>tested by running prediction model while i was playing look at the output and it was accurate in a excellent fashion</li>
    <h5>Rest below</h5>
  </ul>
</p>

# STUNET V1
The neural network worked really well in predicting the keys. e.g it could predict when the car was moving when it was turning , when it crashed (i set as blank input for crashtime and it predicted so in test time). But when left on its own it losses is head as if it had one. But figure of speech. Two major bugs. First I was unable to use pyuserinput module properly to simulate keypress properlyshould not be an issue i guess solvable. Next one i took the training session in a 3rd person view mode this actually introduced a whole new issue. That was the model did not learned anything of the game world i.e roads or traffic it mostly learned about the orientation of the car itself . That means it was good at predicting when the car is in motion and i am controlling it can look at the game view and tell which key i am pressing but when left on its own cant do it properly

<b>OBSERVATION:</b> CNN are good at finding important features. It found the best feature that would help it correlate to the output classes. :-) Well 86% accuracy with 6 hours training on GTX 960 2GB paid off. Though i may try Google Collab. Still working on it.
<b>Youtube link :</b><a href="https://www.youtube.com/watch?v=g2oiNb-_4fQ">https://www.youtube.com/watch?v=g2oiNb-_4fQ</a>
PS:It can get out of corners at sometime. Since i trained it for those data.

<b>Update 2:</b> Fixed the keys. But realised it was using the side walls as guide to move though i never drove like that. Maybe found something in the side wall that correlated with the keys. Hard to say

# STUNET v2
This time we try a new approach. I will split the image for the network to look at. It will not get the image at once put two halves of it. And the network will be forced to not focus on the car anymore and focus on entities like sidelines, markers, white tracks etc hopefully. 
So it worked the model imporoved significantly. The newer model even trained faster on the given dataset. Makes me question i should add more data and diverisify the data. Now first let me explain the bifocal approach. Fancy name i know !!! <br>
<br>
<h4>Bifocal : Nvidias modified Neural network architecture</h4>
First thing look at this article from Nvidia <a href="https://devblogs.nvidia.com/deep-learning-self-driving-cars/">End to end Deep learning Self Driving Car</a> <br>
I used their model for the convolution layers. But where they put the entire image through the sequential network I split the image in 75% from both side. i.e for a width of 400 i will have two image of 250 width starting from both left and right which kinda mimics how our eyes see and percieve depth in world. Then i took the output and simple concatenated it like in inception layers and then flattened and then dense layers. Also i have included more classes at the output layer.<br>
<b>Observation:</b> Everythings improved. Though i did noticed really interesting stuff both in test and train times. During training a batch size of 30 was actually making the accuracy fluctuate at one point around 65% and then suddenly when i decresed the batch size it got worse and then when i increased it to 50 it got better but stuck at 70 and then when finally tuned to 100 gave the best performance. This was obvious since larger batch size averages the differntial over large data and hence will be smaller and more directed towards convergence. During test time that is while driving the data is biased. VERY MUCH BIASED. Even if there is good chance of turning the softmax would give slight higher percentage to forward hence argmax would fail and have forwrad 'W' most times. Issues....
this is the latest update <a href="https://youtu.be/WtuLxI6jLPk">STUNETv2</a><br>
<br>
Next will train same on thirdd person view and have the data shuffled and rid of biases.
