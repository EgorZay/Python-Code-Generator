# Python Code Generator

This is a raw mix-up of how to implement a full-structure LSTM that produces `.py` codes given a random code initialization.

It consists of the following scripts:

* Basic __logger__ (though not a metaclass) that registers different stages of a multi-structural project.

* __getdata__ that parses `dataset` folder and creates a `dataset.dt` file (dictionary class) later used to instantiate inputs.

* __model__ that solely owes a LSTM implementation:
  * First layer: LSTM with an inner dropout of 0.2;
  * Second layer: outer dropout of 0.5;
  * Third layer: LSTM with an inner dropout of 0.2;
  * Last layer: Fully-connected (Dense) layer with softmax activation function.
  
* __train__ that divides inputs into batches and passes them to a created instance of LSTM:
  * Learning_rate (eta): 5e-4;
  * Batch size: 256;
  * Number of epochs: 20.
 
 It also owes two checkpoints that save weights after each epoch and verify an early stopping procedure of 3 epochs tolerance.
 
 * __test__ that is used to produce `.py` codes from different initial states. Each character is an output according to the maximum value of a softmax output.
 
 ____

## Running-yourself procedure:

- If you are willing to test it out yourself, then first of all you should run `getdata.py` by adding the following code in the end of the script:
 
       if __name__ == '__main__':
          gt = GetData()
          gt.get_dataset()
       
Provided you have `dataset` folder with `.py` scripts, it will therefore parse the folder and create a `dataset.dt` database inside.

- You should then add the following code chunk in the end of `train.py`:
    
       if __name__ == '__main__':
          tr = Train()
          tr.fit()
       

Which will train a model while saving weigths after a successful epoch. In the end, the model is saved in `model/model.h5`.

- Finally, you run `test.py` by adding the next code chunk:

       if __name__ == '__main__':
          t = Test()
          model = t.load_model()
          t.predict(model, script_len=50)
  
Which should provide you with four brand-new-meaningless scripts not following a PEP-8 convention.

____

## Aftermath

Due to lack of computational power I could not go further 20 epochs of training having a corpus of ~500k characters. It thus produced absolutely incoherent text not following basic coding structure and indentation. Also, what I have used as a model was a simple LSTM-128 (though inner dropout was present) that partly caused a failure during a `test.py` stage.

I encourage anyone with enough computational power to test it providing a sufficient feedback through issuing.
