# ML Student Classification

reproducing jadud et al's machine learning student classification study with data from U Toronto's first year programming class

### Dependencies
* numpy
* scipy
* sklearn
* tensorflow
* WEKA

### **Data Processing** 

Set dataset name in 

	data_util/dataset_name.txt

To do the primary import from progsnap:  

	./1_build_features

further work with the data is the loaded from an intermediate .arff file for speed (reading the progsnap format is slow ~ lots of dirs)

data_util.processor.DataProcessor converts progsnap to .arff and manipulate, filter and clean progsnap data in various ways.

Progsnap features are also convertable to vector format

*See convert_data.py for use of DataProcessor* 

*NOTE: code_data.csv has been truncated to exclude code for size 

### Multilayer Perceptron

multitron.py

Using as few layers as possible turns out to be ideal for this data set.

[credit](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py)

### Model Validation

	python validate_models.py

To run weka and the perceptron on the data as defined in convert_data.py
