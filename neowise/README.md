
# neowise

### A Deep Learning library built from scratch with Python and NumPy
![logo](/neowise.png)

### pip install 
`pip install neowise`

### [Documentation](https://github.com/pranavsastry/neowise/blob/master/DOCUMENTATION.md) 

### Features of *neowise*

 - Get summary of your model, by calling `model.summary`
  ![summary](Visuals/summary.png)<br/>
 - Save your model in a .h5 file using `model.save_model`<br/>
 - Load your saved model with `model.load_model`<br/>
 - Train your model with less than 10 lines of code (*excluding data processing*), with a simple API<br/>
 - Test your model with `model.test`<br/>
   ![test](Visuals/test.png)
 - Plot static graphs of Cost and Accuracy using `model.plot`<br/>
   ![costs](/Visuals/costs.png)<br/>
   ![accuracy](Visuals/accuracy.png)<br/>
 - Train using optimizers such as Gradient Descent, Momentum, RMSprop, Adam, Batch Gradient Descent and Stochastic Gradient Descent<br/>
 - Train using Dropout regularization for deeper networks<br/>
 - While, training the models, keep track of your model's progress through tqdm's progress bar<br/>
   ![fit](Visuals/fit.png)<br/>
 - Create animated graphs of Cost and Accuracy with `model.plot` and set `animate=True` to save images of plots which can then be fed to a GIF creator to create an animated GIF <br/>
   ![costs_gif](Visuals/costs.gif)<br/>
   ![accu_gif](Visuals/accu.gif)<br/>
