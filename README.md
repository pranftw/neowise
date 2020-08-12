
# neowise <br/>
[![GitHub license](https://img.shields.io/github/license/pranavsastry/neowise)](https://github.com/pranavsastry/neowise/blob/master/LICENSE)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fpranavsastry%2Fneowise)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fpranavsastry%2Fneowise)

### A Deep Learning library built from scratch with Python and NumPy
![logo](/neowise.png)

### pip install 
`pip install neowise`

### [Documentation](https://github.com/pranavsastry/neowise/blob/master/DOCUMENTATION.md) 

### Features of *neowise*

 - Get summary of your model, by calling `model.summary`
  ![summary](neowise/Visuals/summary.png)<br/>
 - Save your model in a .h5 file using `model.save_model`<br/>
 - Load your saved model with `model.load_model`<br/>
 - Train your model with less than 10 lines of code (*excluding data processing*), with a simple API<br/>
 - Test your model with `model.test`<br/>
   ![test](neowise/Visuals/test.png)
 - Plot static graphs of Cost and Accuracy using `model.plot`<br/>
   ![costs](neowise/Visuals/costs.png)<br/>
   ![accuracy](neowise/Visuals/accuracy.png)<br/>
 - Train using optimizers such as Gradient Descent, Momentum, RMSprop, Adam, Batch Gradient Descent and Stochastic Gradient Descent<br/>
 - Train using Dropout regularization for deeper networks<br/>
 - While, training the models, keep track of your model's progress through tqdm's progress bar<br/>
   ![fit](neowise/Visuals/fit.png)<br/>
 - Create animated graphs of Cost and Accuracy with `model.plot` and set `animate=True` to save images of plots which can then be fed to a GIF creator to create an animated GIF <br/>
   ![costs_gif](neowise/Visuals/costs.gif)<br/>
   ![accu_gif](neowise/Visuals/accu.gif)<br/>
