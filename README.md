# style-transfer

Porting the style of one image onto another using neural networks, gardient descent and some cleverly chosen loss functions. The code is largely taken from [this blogpost](https://keras.io/examples/generative/neural_style_transfer/).


![](me.jpg) ![](https://upload.wikimedia.org/wikipedia/commons/thumb/archive/7/7d/20071024210407!Tab_plus.svg/120px-Tab_plus.svg.png) ![](flower.jpg) ![](https://etc.usf.edu/clipart/41700/41709/fc_equalto_41709_mth.gif) ![](2000.png)

After a while, the red colour dominates and overwhelms any texture of the style image. I added an additional loss function to penalise colour change from the reference image, resulting in the following output:

![](400_with_loss.png)
