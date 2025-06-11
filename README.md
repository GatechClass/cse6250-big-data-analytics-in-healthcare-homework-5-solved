# cse6250-big-data-analytics-in-healthcare-homework-5-solved
**TO GET THIS SOLUTION VISIT:** [CSE6250: Big Data Analytics in Healthcare Homework 5 Solved](https://mantutor.com/product/cse6250-big-data-analytics-in-healthcare-homework-5-solved/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;12652&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE6250: Big Data Analytics in Healthcare Homework 5 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
<h1>Overview</h1>
Neural Networks and Deep Learning are becoming more popular and widely applied to many data science domain including healthcare applications. In this homework, you will implement various types of neural network with clinical data. For this homework, <strong>Python </strong>programming will be required. See the attached skeleton code as a starting point for the programming questions. Also, you need to <strong>make a single PDF file (</strong><em>homework5 </em><em>answer.pdf</em><strong>) </strong>of a compiled document for non-programming questions. You can use the attached L<sup>A</sup>TEXtemplate.

It is highly recommended to complete <a href="http://www.sunlab.org/teaching/cse6250/fall2018/dl/dl-setup.html#framework">Deep Learning Lab</a> first if you have not finished it yet. (PyTorch version could be different, but most of parts are same.)

<h1>Python and dependencies</h1>
In this homework, we will work with <strong>PyTorch </strong>1.0 on Python 3.6 environment. If you do not have a python distribution installed yet, we recommend installing <a href="https://docs.continuum.io/anaconda/install">Anaconda</a> (or miniconda) with Python 3.6. We provide <em>homework5/environment.yml </em>which contains a list of libraries needed to set environment for this homework. You can use it to create a copy of conda ‘environment’. Refer to <a href="http://conda.pydata.org/docs/using/envs.html#use-environment-from-file">the documantation</a> for more details.

<table width="633">
<tbody>
<tr>
<td width="65">conda</td>
<td width="40">env</td>
<td width="528">create -f environment.yml</td>
</tr>
</tbody>
</table>
If you already have your own Python development environment (it should be Python 3.6), please refer to this file to find necessary libraries, which is used to set the same coding/grading environment.

<h1>Content Overview</h1>
homework5

|– code

| |– etl_mortality_data.py

| |– mydatasets.py

| |– mymodels.py

| |– plots.py

| |– tests

| | |– test_all.py

| |– train_seizure.py

| |– train_variable_rnn.py

| |– utils.py

|– data

| |– mortality

| | |– test

| | | |– ADMISSIONS.csv

| | | |– DIAGNOSES_ICD.csv

| | | |– MORTALITY.csv

| | |– train

| | | |– ADMISSIONS.csv

<table width="633">
<tbody>
<tr>
<td width="633">| | | |– DIAGNOSES_ICD.csv

| | | |– MORTALITY.csv

| | |– validation

| |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– ADMISSIONS.csv

| |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– DIAGNOSES_ICD.csv

| |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– MORTALITY.csv | |– seizure

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– seizure_test.csv

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– seizure_train.csv

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |– seizure_validation.csv

|– environment.yml

|– homework5.pdf

|– homework5_answer.tex
</td>
</tr>
</tbody>
</table>
<h1>Implementation Notes</h1>
Throughout this homework, we will use <a href="https://en.wikipedia.org/wiki/Cross_entropy"><em>Cross Entropy</em></a> loss function, which has already been applied in the code template. <strong>Please note that this loss function takes activation logits as its input, not probabilities. Therefore, your models should not have a softmax layer at the end. </strong>Also, you may need to control the training parameters declared near the top of each main script, e.g., the number of epochs and the batch size, by yourself to achieve good model performances.

You will submit a trained model for each type of neural network, which will be saved by the code template, and they will be evaluated on a hidden test set in order to verify that you could train the models. Please make sure that your saved/submitted model files match your model class definitions in your source code. <strong>You will get 0 score for each part in either case you do not submit your saved model files or we cannot use your model files directly with your code.</strong>

<h2>Unit tests</h2>
Some unit tests are provided in <em>code/tests/test all.py</em>. You can run all tests, <strong>in your </strong>code <strong>directory</strong>, simply by

python -m pytest

<strong>NOTE: the primary purpose of these unit tests is just to verify the return data types and structures to match them with the grading system.</strong>

<h1>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Epileptic Seizure Classification [50 points]</h1>
For the first part of this homework, we will use Epileptic Seizure Recognition Data Set which is originally from <a href="http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition">UCI Machine Learning Repository</a><a href="http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition">.</a> You can see the three CSV files under the folder <em>homework5/data/seizure</em>. Each file contains a header at the first line, and each line represent a single EEG record with its label at the last column (Listing 1). Refer to the link for more details of this dataset.

X1,X2,…,X178,y

-104,-96,…,-73,5

-58,-16,…,123,4 …

Listing 1: Example of seizure data

Please start at <em>train seizure.py</em>, which is a main script for this part. If you have prepared the required Python environment, you can run all procedures simply by

python train_seizure.py

<h2>1.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data Loading [5 points]</h2>
First of all, we need to load the dataset to train our models. Basically, you will load the raw dataset files, and convert them into a PyTorch <a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset">TensorDataset</a> which contains data tensor and target (label) tensor. Since each model requires a different shape of input data, you should convert the raw data accordingly. Please look at the code template, and you can complete each type of conversion at each stage as you progress.

<ol>
<li>Complete load seizure dataset in mydatasets.py [5 points]</li>
</ol>
<h2>1.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Multi-layer Perceptron [15 points]</h2>
We start with a simple form of neural network model first, Multi-layer Perceptron (MLP).

<ol>
<li>Complete a class MyMLP in py, and implement 3-layer MLP similar to the Figure 1. Use a hidden layer composed by 16 hidden units, followed by a sigmoid activation function. [3 points]</li>
<li>Calculate the number of ”trainable” parameters in the model with providing the calculation details. How many floating-point computation will occur <strong>APPROXIMATELY </strong>when a new single data point comes in to the model? <strong>You can make your own assumptions on the number of computations made by each elementary arithmetic, e.g., add/subtraction/multiplication/division/negation/exponent take 1 operation, etc. </strong>[5 points]</li>
</ol>
Figure 1: An example of 3-layer MLP

(a) Loss curves&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) Accuracy curves

Figure 2: Learning curves

It is important to monitor whether the model is being trained well or not. For this purpose, we usually track how some metric values, loss for example, change over the training iterations.

<ol>
<li>Complete a function plot learning curves in py to plot loss curves and accuracy curves for training and validation sets as shown in Figure 2a. You should control the number of epochs (in the main script) according to your model training. Attach the plots for your MLP model in your report. [2 points]</li>
</ol>
After model training is done, we also need to evaluate the model performance on an unseen (during the training procedure) test data set with some performance metrics. We will use <a href="https://en.wikipedia.org/wiki/Confusion_matrix"><em>Confusion Matrix</em></a> among many possible choices.

<ol start="3">
<li>Complete a function plot confusion matrix in py to make and plot a confusion matrix for test set similar to Figure 3. Attach the plot for your MLP model in your report. [2 points]</li>
</ol>
Figure 3: An example of confusion matrix

You have implemented a very simple MLP model. Try to improve it by changing model parameters, e.g., a number of layers and units, or applying any trick and technique applicable to neural networks.

<ol>
<li>Modify your model class MyMLP in py to improve the performance. It still must be a MLP type of architecture. Explain your architecture and techniques used. Briefly discuss about the result with plots. [3 points]</li>
</ol>
<h2>1.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Convolutional Neural Network (CNN) [15 points]</h2>
Our next model is CNN. Implement a CNN model constituted by a couple of convolutional layers, pooling layer, and fully-connected layers at the end as shown in Figure 4.

<ol start="2">
<li>Complete MyCNN class in mymodels.py. Use two convolutional layers, one with 6 filters of the kernel size 5 (stride 1) and the other one with 16 filters with the kernel size 5 (stride 1), and they must be followed by Rectified Linear Unit (ReLU) activation. Each convolutional layer (after ReLU activation) is followed by a max pooling layer with the size (as well as stride) of 2. There are two fully-connected (aka dense) layer, one with 128 hidden units followed by ReLU activation and the other one is the output layer that has five units.</li>
</ol>
[5 points]

<ol>
<li>Calculate the number of ”trainable” parameters in the model with providing the calculation details. How many floating-point computation will occur <strong>APPROXIMATELY </strong>when a new single data point comes in to the model? <strong>You can make your own assumptions on the number of computations made by each elementary arithmetic, e.g., add/subtraction/multiplication/division/negation/exponent take 1 operation, etc. </strong>[5 points]</li>
</ol>
Figure 4: CNN architecture to be implemented. <em>k </em>Conv1d (size d) means there are <em>k </em>numbers of Conv1d filters with a kernel size of <em>d</em>. Each convolution and fully-connected layer is followed by ReLU activation function except the last layer.

Figure 5: A many-to-one type of RNN architecture to be implemented.

<ol>
<li>Plot and attach the learning curves and the confusion matrix for your CNN model in</li>
</ol>
your report. [2 points]

Once you complete the basic CNN model, try to improve it by changing model parameters and/or applying tricks and techniques.

<ol>
<li>Modify your model class MyCNN in py to improve the performance. It still must be a CNN type of architecture. Explain your architecture and techniques used. Briefly discuss about the result with plots. [3 points]</li>
</ol>
<h2>1.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Recurrent Neural Network (RNN) [15 points]</h2>
The final model you will implement on this dataset is RNN. For an initial architecture, you will use a Gated Recurrent Unit (GRU) followed by a fully connected layer. Since we have a sequence of inputs with a single label, we will use a many-to-one type of architecture as it is shown in Figure 5.

<ol>
<li>Complete MyRNN class in py. Use one GRU layer with 16 hidden units. There should be one fully-connected layer connecting the hidden units of GRU to the output units. [5 points]</li>
<li>Calculate the number of ”trainable” parameters in the model with providing the calculation details. How many floating-point computation will occur <strong>APPROXIMATELY </strong>when a new single data point comes in to the model? <strong>You can make your own assumptions on the number of computations made by each elementary arithmetic, e.g., add/subtraction/multiplication/division/negation/exponent take 1 operation, etc. </strong>[5 points]</li>
<li>Plot and attach the learning curves and the confusion matrix for your RNN model in your report. [2 points]</li>
<li>Modify your model class MyCNN in py to improve the performance. It still must be a RNN type of architecture. Explain your architecture and techniques used. Briefly discuss about the result with plots. [3 points]</li>
</ol>
<h1>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Mortality Prediction with RNN [45 points + 5 bonus]</h1>
In the previous problem, the dataset consists of the same length sequences. In many realworld problems, however, data often contains variable-length of sequences, natural language processing for example. Also in healthcare problems, especially for longitudinal health records, each patient has a different length of clinical records history. In this problem, we will apply a recurrent neural network on variable-length of sequences from longitudinal electronic health record (EHR) data. Dataset for this problem was extracted from MIMICIII dataset. You can see the three sub-folders under the folder <em>homework5/data/mortality</em>, and each folder contains the following three CSV files.

<ul>
<li><strong>csv</strong>: A pre-processed label file with the following format with a header. SUBJECT ID represents a unique patient ID, and MORTALITY is the target label for the patient. <strong>For the test set, all labels are -1, and you will submit your predictions on Kaggle competition at the end of this homework.</strong></li>
</ul>
<table width="593">
<tbody>
<tr>
<td width="593">SUBJECT_ID,MORTALITY

123,1

456,0 …
</td>
</tr>
</tbody>
</table>
<ul>
<li><strong>csv</strong>: A filtered MIMIC-III ADMISSION table file. It has patient visits (admissions) information with unique visit IDs (HADM ID) and dates (ADMITTIME)</li>
<li><strong>DIAGNOSES ICD.csv</strong>: A filtered MIMIC-III DIAGNOSES ICD table file. It contains diagnosis information as ICD-9 code given to patients at the visits.</li>
</ul>
Please start at <em>train variable rnn.py</em>, which is a main script for this part. You can run all procedures simply by

python train_variable_rnn.py

<strong>once </strong>you complete the required parts below.

Again, you will submit a trained model, and it will be evaluated on a hidden test set in order to verify that you could train the models.

<h2>2.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Preprocessing [10 points]</h2>
You may already have experienced that preprocessing the dataset is one of the most important part of data science before you apply any kind of machine learning algorithm. Here, we will implement a pipeline that process the raw dataset to transform it to a structure that we can use with RNN model. You can use typical Python packages such as Pandas and Numpy.

<strong>Simplifying the problem, we will use the main digits, which are the first 3 or 4 alphanumeric digits prior to the decimal point, of ICD-9 codes as features in this homework.</strong>

The pipeline procedure is very similar to what you have done so far in the past homework, and it can be summarized as follows.

<ol>
<li>Loading the raw data files.</li>
<li>Group the diagnosis codes given to the same patient on the same date.</li>
<li>Sorting the grouped diagnoses for the same patient, in chronological order.</li>
<li>Extracting the main code from each ICD-9 diagnosis code, and converting them into unique feature ids, 0 to <em>d </em>− 1, where <em>d </em>is the number of unique main -digits codes.</li>
<li>Converting into the final format we want. Here we will use a List of (patient) List of (code) List as our final format of the dataset.</li>
</ol>
The order of some steps can be swapped for implementation convenience. Please look at the main function in etl mortality data.py first, and do the following tasks.

<ol>
<li>Complete convert icd9 function in etl mortality data.py. Specifically, extract the the first 3 or 4 alphanumeric characters prior to the decimal point from a given ICD-9 code. Please refer to the <a href="https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/HospitalQualityInits/Downloads/HospitalAppendix_F.pdf">ICD-9-CM format</a> for the details, and note that there is no actual decimal point in the representation by MIMIC-III. [2 points]</li>
<li>Complete build codemap function in etl mortality data.py. Basically, it is very similar to what you did in Homework 1 and is just constructing a unique feature ID map from the main-digits of the ICD-9 codes. [1 point]</li>
<li>Complete create dataset function in etl mortality data.py referring to the sum-</li>
</ol>
marized steps above and the TODO comments in the code. [7 points]

For example, if there are two patients with the visit information below (in an abstracted format),

SUBJECT_ID,ADMITTIME,ICD9_CODE

1, 2018-01-01, 123

1, 2018-01-01, 234

2, 2013-04-02, 456

2, 2013-04-02, 123

3, 2018-02-03, 234

2, 2013-03-02, 345

2, 2013-05-03, 678

2, 2013-05-03, 234

3, 2018-09-07, 987

the final visit sequence dataset should be

[[[0, 1]], [[3], [0, 2], [4, 1]], [[1], [5]]]

with the order of [Patient 1, Patient 2, Patient 3] and the code map {123: 0, 234: 1, 456: 2, 345: 3, 678: 4, 987: 5}.

Once you complete this pre-processing part, you should run it to prepare data files required in the next parts.

python etl_mortality_data.py

<h2>2.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Creating Custom Dataset [15 points]</h2>
Next, we will convert the pre-processed data in the previous step into a custom type of PyTorch Dataset. When you create a custom (inherited) PyTorch Dataset, you should override at least lenand getitem &nbsp;methods. len &nbsp;method just returns the size of dataset, and getitem&nbsp;actually returns a sample from the dataset according to the given index. Both methods have already been implemented for you. Instead, you have to complete the constructor of VisitSequenceWithLabelDataset class to make it a valid ’Dataset’. We will store the labels as a List of integer labels, and the sequence data as a List of matrix whose <em>i</em>-th row represents <em>i</em>-th visit while <em>j</em>-th column corresponds to the integer feature ID <em>j</em>. For example, the visit sequences we obtained from the previous example will be transformed to matrices as follows.

<sup></sup>0&nbsp;&nbsp; 0&nbsp;&nbsp; 0&nbsp;&nbsp; 1&nbsp;&nbsp; 0&nbsp;&nbsp; 0<sup></sup>

P1 = <em>&nbsp;, </em>P2 = 1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0&nbsp;&nbsp; 1&nbsp;&nbsp; 0&nbsp;&nbsp; 0&nbsp;&nbsp;&nbsp;&nbsp; 0<em>, </em>P3 =&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>.</em>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 

0&nbsp;&nbsp; 1&nbsp;&nbsp; 0&nbsp;&nbsp; 0&nbsp;&nbsp; 1&nbsp;&nbsp; 0

<ol>
<li>Complete calculate num features function and VisitSequenceWithLabelDataset</li>
</ol>
class in mydatasets.py. [5 points]

PyTorch DataLoader will generate mini-batches for you once you give it a valid Dataset, and most times you do not need to worry about how it generates each mini-batch if your data have fixed size. In this problem, however, we use variable size (length) of data, e.g., visits of each patient are represented by a matrix with a different size as in the example above. Therefore, we have to specify how each mini-batch should be generated by defining collate fn, which is an argument of DataLoader constructor. Here, we will create a custom collate function named visit collate fn, which creates a mini-batch represented by a 3D Tensor that consists of matrices with the same number of rows (visits) by padding zero-rows at the end of matrices shorter than the largest matrix in the mini-batch. Also, the order of matrices in the Tensor must be sorted by the length of visits in descending order. In addition, Tensors contains the lengths and the labels are also have to be sorted accordingly.

Please refer to the following example of a tensor constructed by using matrices from the previous example.

<table width="206">
<tbody>
<tr>
<td width="103">0

<em>T</em>[0<em>,</em>:<em>,</em>:] = 1



0
</td>
<td width="21">0 0 1</td>
<td width="21">0 1 0</td>
<td width="21">1 0 0</td>
<td width="21">0 0 1</td>
<td width="18">0

0



0
</td>
</tr>
<tr>
<td width="103">0

<em>T</em>[1<em>,</em>:<em>,</em>:] = 0



0
</td>
<td width="21">1 0 0</td>
<td width="21">0 0 0</td>
<td width="21">0 0 0</td>
<td width="21">0 0 0</td>
<td width="18">0

1



0
</td>
</tr>
<tr>
<td width="103">1

<em>T</em>[2<em>,</em>:<em>,</em>:] = 0



0
</td>
<td width="21">1 0 0</td>
<td width="21">0 0 0</td>
<td width="21">0 0 0</td>
<td width="21">0 0 0</td>
<td width="18">0

0



0
</td>
</tr>
</tbody>
</table>
where <em>T </em>is a Tensor contains a mini-batch size of batch size × max length × num features, and batch size=3, max length=3, and num features=6 for this example.

<ol>
<li>Complete visit collate fn function in mydatasets.py. [10 points]</li>
</ol>
<h2>2.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Building Model [15 points]</h2>
Now, we can define and train our model. Figure 6 shows the architecture you have to implement first, then you can try your own model.

Figure 6: A RNN architecture to be implemented with variable-length of sequences.

<ol>
<li>Complete MyVariableRNN class in py. First layer is a fully-connected layer with 32 hidden units, which acts like an embedding layer to project sparse high-dimensional inputs to dense low-dimensional space. The projected inputs are passed through a GRU layer that consists of 16 units, which learns temporal relationship. It is followed by output (fully-connected) layer constituted by two output units. In fact, we can use a single output unit with a sigmoid activation function since it is a binary classification problem. However, we use a multi-class classification setting with the two output units to make it consistent with the previous problem for reusing the utility functions we have used. Use tanh activation funtion after the first FC layer, and remind that you should not use softmax or sigmoid activation after the second FC (output) layer since the loss function will take care of it. Train the constructed model and evaluate it. Include all learning curves and the confusion matrix in the report. [10 points]</li>
<li>Modify your model class MyVariableRNN in py to improve the performance. It still must be a RNN type that supports variable-length of sequences. Explain your architecture and techniques used. Briefly discuss about the result with plots. [5 points]</li>
</ol>
<h2>2.4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Kaggle Competition [5 points + 5 bonus]</h2>
<strong>NOTE: NO late submission will be accepted for this Kaggle competition.</strong>

You should be familiar with Kaggle, which is a platform for predictive modeling and analytics competitions, that you used in the past homework. You will compete with your class mates using the result by your own model that you created in Q2.3.b.

Throughout this homework, we did not use sigmoid activation or softmax layer at the end of the model output since CrossEntropy loss applies it by itself. For Kaggle submission, however, we need to have a soft-label, probability of mortality, for each patient. For this purpose, you have to complete a function predict mortality function in the script to evaluate the test set and collect the probabilities.

<ol>
<li>Complete predict mortality function in train variable rnn.py, submit the generated predictions (output/mortality/my predictions.csv) to the Kaggle competition, and achieve AUC <em>&gt; </em>0.6. [5 points]</li>
</ol>
<strong>NOTE: it seems Kaggle accepts 20 submissions at maximum per day for each user.</strong>

Submit your <strong>soft-labeled </strong>CSV prediction file (MORTALITY is in the range of [0,1]) you generate to this <a href="https://www.kaggle.com/t/752ea84a6fb846f6a8d081a15502a108">Kaggle competition</a><a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> (<strong>you can join the competition via this link only</strong>) created specifically for this assignment to compete with your fellow classmates. A <strong>gatech.edu </strong>email address is required to participate; follow the sign up direction using your GT email address, or if you already have an account, change the email on your existing account via your profile settings so that you can participate. <strong>Make sure your display name (not necessarily your username) matches your GT account username, e.g., san37, </strong>so that your Kaggle submission can be linked to the grading system.

Evaluation criteria is AUC, and the predicted label should be a soft label, which represents the possibility of mortality for each patient you predict. The label range is between 0 and 1. 50% of the data is used for the public leaderboard where you can receive feedback on your model. The final private leaderboard will use the remaining 50% of the data and will determine your final class ranking.

Score at least 0.6 AUC to receive 5 points of credit. Additional bonus points can be received for your performance according to the following:

&nbsp;

&nbsp;
