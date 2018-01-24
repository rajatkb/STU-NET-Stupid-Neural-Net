# STU-NET-Stupid-Neural-Net
STU-NET as the acronym goes is a really bad attempt at making race car A.I . So the attempt is to have the car drift , slide and move with really high speed. Because mainstream self driving a.i is mehh!! 

<h3>
  How i made the model
</h3>
<p>
  <ul>
    <li> Got a lot of hints from Sentdex GTAV A.I tutorial aka CHARLES THE A.I , got scripts like screen capture and keys capture from there</li>
    <li>Next Got some training data by playing the game. The model was quite simple for thic CNN => One hot encoding of movement key classes. So made the gameplay adat according to that</li>
    <li>trained on my own model architecture</li>
    <li>tested by running prediction model while i was playing look at the output and it was accurate in a excellent fashion</li>
    <h5>Rest below</h5>
  </ul>
</p>

# STUNET V1
The neural network worked really well in predicting the keys. e.g it could predict when the car was moving when it was turning , when it crashed (i set as blank input for crashtime and it predicted so in test time). But when left on its own it losses is head as if it had one. But figure of speech. Two major bugs. First I was unable to use pyuserinput module properly to simulate keypress properlyshould not be an issue i guess solvable. Next one i took the training session in a 3rd person view mode this actually introduced a whole new issue. That was the model did not learned anything of the game world i.e roads or traffic it mostly learned about the orientation of the car itself . That means it was good at predicting when the car is in motion and i am controlling it can look at the game view and tell which key i am pressing but when left on its own cant do it properly

<b>OBSERVATION:</b> CNN are good at finding important features. It found the best feature that would help it correlate to the output classes. :-) Well 86% accuracy with 6 hours training on GTX 960 2GB paid off. Though i may try Google Collab. Still working on it.
<b>Youtube link :</b><a href="https://www.youtube.com/watch?v=g2oiNb-_4fQ">https://www.youtube.com/watch?v=g2oiNb-_4fQ</a>
