```
        "gilga.msh inc.
         _________           ___    ___            ________         ___       __  
       /"         `\\      /   \\  /   \\         /         \\    //  \\\\   /   \\ 
      //    ________/     /     \\/     \\      //    ..-.___\\   '      \\_/     . 
     /    /  ______       ;              \\     /    //    \"\"   |               |
    \\    \ |____  \    /    /\\    /\\   \\   \\    \\    ___    |       _       |
     \\    \____/   /  /    /  \\__/  \\   \\   \\    '--'   //   '      / \\     ' 
      \\ __________/   \\__/          \\___/     \\_________/      \\___/   \\___/  
```

# Check out full website
**Live Demo**: [Click here to see the current online version](https://glmch-ensisa.alwaysdata.net/)

# Hand Written Digit Recognition
This project uses **Convolutional Neural Networks (CNNs)** to recognize handwritten digits and full mathematical expressions. It started as a simple digit recognizer and was later extended to handle entire handwritten expressions.

## Features
- Predict individual handwritten digits using a trained CNN.
- Recognize full mathematical expressions from handwritten input.
- Interactive web interface powered by **Flask**, **HTML**, **CSS**, **JavaScript**, and **jQuery**.
- Real-time predictions.

# Usage
After running train.py, a mnist_cnn.pth file is created with the model's parameters, as well as the training data if not already installed. Use app.py to run a server and try it out for yourself, note that you can plot what's being sent to the CNN by uncommenting the matplotlib section.

# Problems
You may notice a lot of unnecessary files, they were used to train the first version on the MNIST dataset, to recognize only one digit. I chose to keep them to be able to retrain that model, and maybe learn how to merge the code better in the future.

