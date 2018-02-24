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

So it worked the model imporoved significantly. The newer model even trained faster on the given dataset. Makes me question i should add more data and diverisify the data. Now first let me explain the bifocal approach. Fancy name i know !!!

<h4>Bifocal : Nvidias modified Neural network architecture</h4>
First thing look at this article from Nvidia <a href="https://devblogs.nvidia.com/deep-learning-self-driving-cars/">End to end Deep learning Self Driving Car</a> I used their model for the convolution layers. But where they put the entire image through the sequential network I split the image in 75% from both side. i.e for a width of 400 i will have two image of 250 width starting from both left and right which kinda mimics how our eyes see and percieve depth in world. Then i took the output and simple concatenated it like in inception layers and then flattened and then dense layers. Also i have included more classes at the output layer.

<b>OBSERVATION:</b> Everythings improved. Though i did noticed really interesting stuff both in test and train times. During training a batch size of 30 was actually making the accuracy fluctuate at one point around 65% and then suddenly when i decresed the batch size it got worse and then when i increased it to 50 it got better but stuck at 70 and then when finally tuned to 100 gave the best performance. This was obvious since larger batch size averages the differntial over large data and hence will be smaller and more directed towards convergence. During test time that is while driving the data is biased. VERY MUCH BIASED. Even if there is good chance of turning the softmax would give slight higher percentage to forward hence argmax would fail and have forwrad 'W' most times. Issues....
this is the latest update Youtube: <a href="https://youtu.be/WtuLxI6jLPk">STUNETv2</a><br>
<br>
Next will train same on third person view and have the data shuffled and rid of biases.  
<b>Update :</b> Well i did that. I updated my code to have the intake or the data collection operation regulate the intake number of forward vs other movement. Earlier the data was heavily biased towards forward movement. Now with that bias removed the data is more unbiased hence the model will try to fit it to the various scenarios of the data. The model is doing good compared to previous one with just mere 10k data. We nee atleast 100k.  

<b>Next :</b> We add more data and Will be shifting to pretrained YOLO model possibly for object detection and use those input as the input for driving movement along with the input of bifocal_nvidia. The objective here is to perfect the instantaneous drive decisions so that later on when the model is extended with RNN and Q-Learning it will atleast have good features to take its decision or learn from past values i.e Sequence. 

<b>ISSUE:</b> 
<img src="https://i.imgur.com/3QpGfya.png" />
This model is not being effective on the road. Its erratic. It acts really well at some points. At the other it doesnt knows. So the perception problem needs to be solved.  

# STUNET v3  
The only major changes are well using a bit higher res image. This time trying to fit a 100x200 image into the layers and see if that yields any nice results. Apart from that i missed normalization in the beggining of the input. Added that so now in the same 2nd layer the features are more prominent. Using fully dense layers now with no dropouts what so ever. Some visualized layers. Features are more prominent now. But the game world lacks consistency for the model to capture especially in the track lines. I am shifting between 3rd person and first person view without any major gain. 

<b>Upddate 7 Feb :</b>  
<img src="https://i.imgur.com/AnW27Zj.png"/>  
  It did learn something. Need to figure out proper image size for maximising feature extraction.

<b>Update :</b> More data improved it maybe and here is the visualization of the lower levels  
Youtube link: <a src="https://www.youtube.com/watch?v=a7gLpCjQMUk">STUNETv3 improved</a>  
Ohh look at this too better filter maybe  

<img src="https://i.imgur.com/z76WP21.png" />  
  So this one is my final attempt at this model  

Before shifting to a new model i would increase the data and try to see whether it drives better or not. Whats kind of worrying to me is the speed at which it is training and converging. It converges really fast. So it kind of makes me wonder is it finding shortcuts in the data. Though i dont mind shortcuts provided it generalizes well , but then again its not generalizing well. I just have to test with more training data with approx. of 100k frames of data. Currently i am training with 26k frames , which takes only about few minutes to train.  
Sentdex did 18 hours of training on a TITAN X to get the same kind of accuracy after his initiall attempts with AlexNet. And here i kind of have almost similar result compared to his beggining versions with this model and more controlled data. So actually this is actually kind of worrying. So NEED MORE DATA !!..  

<img src="https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png" />  
  Ya this is the nvidia's architecture . Quite small and simple and it works way to good.  
  
<img src="https://i.imgur.com/reAD4Ph.png" />
  And this is my own modified version.

I know nothing special here but then again maybe this wierd model modfication is maybe why i can see better result in smaller but proper training sets. The next movie will to add a Eulerian distance in the concatenation point. Hence essentially creating a bottleneck which will i think may turn into a meaussured metric of how much the car should turn left or right. But i think this will only work with a recurrent network since this meassured metric will change in instants and hence i need the model to consider past values and then take turns. So next step EULERIAN BIFOCAL.  

# STUNET v4

And the idea worked i think. The idea of introducing a bottleneck or rather the original steer angle output in between the softmax layer and the fully connected one. It not more (2 * CNN ==> F.C ==> softmax) now it is (2 * CNN ==> F.C ==> one neuron ==> F.C ==? softmax) . This model took its fair amount of time. Since in iinnitial attempts the model was not training because it was getting stuck at some local minima repeatedly. No matter what or how much data i give whatever parameter except batch size (since hardware limitation) i chnage it was not helping. I was looking around internet and find this advice somewhere from some paper by Bengio of using Batch Norm. Though the reason was different but i attempted it and it worked the network trained. And it was a steady training more over. I trained it for 4 hours approx on GTX 960 to get an acuracy of 86% and loss of 0.3. Furthur training should improve the network. The performance is better provided now it can make really great minute changes. Now the data really controls the driving. Its highly dependent on data. I have chnage the train data by driving differently and it picked it up really well. So yes this time the model works. The next challenge is to make the model understand the speed of the car. Yes <b>SPEED</b> !! . High speed is not good at turns and low speed when moving forward it needs to figure out on its own how to do that. Since the system is end to end hence less intervention and more the network needs to figure out. Lets attempt at it .....

new stunet v4 drive video : <a src="https://www.youtube.com/watch?v=wwdEvaPljBQ&feature=youtu.be">Youtube link</a>


