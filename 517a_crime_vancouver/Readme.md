##	Crime in Vancouver

In this project, we collected crime data records from the Vancouver Open Data Catalogue as instructed by Kaggle.
Th dataset can be forund here: https://www.kaggle.com/wosaku/crime-in-vancouver/data

The original data has 530,652 records from 2003-01-01 to 2017-07-13 with the following features:

#### Data Schema:

Columns:
TYPE, YEAR, MONTH, DAY, HOUR, MINUTE, HUNDRED_BLOCK, NEIGHBOURHOOD, X, Y, LATITUDE, LONGITUDE.

The record consists of 9 different types of crimes in 24 neighborhoods and 21193 street names. 

In this project we train different classifiers on the crime data to identify crimes relating to collisions or fatality (Homicide and Pedestrian Struck with Fatality or Injury) only. 

#### Initial Data Analysis:

**data_analysis.py**

This files provides all the necessary functions to generate initial results from the dataset.

Running the analysis function inside data_analysis.py generates the different types of columns and the different types of criminal activities reported.



#### Data preprocessing
From the 530652 records, we identified 22141 records that are related to collision. We randomly selected 27859 records to create a training set with 50000 records.
Computing statistics on the dataset, we noticed that the distribution of crimes per day follows a normal distribution with a mean around 95 and a few outliers over 600. The time series plot demonstrates how the number of criminal activities varied within 2 standard deviations. Notice that there is a huge spike between the years 2011 and 2012 suggesting that there was a disproportionately more crime occuring during in that period.
![Alt text](https://raw.githubusercontent.com/puchiha/crime_data_analysis/master/517a_crime_vancouver/plots/dist_crimes_per_day.png)
![Alt text](https://raw.githubusercontent.com/puchiha/crime_data_analysis/master/517a_crime_vancouver/plots/time_series_analysis.png)


The processed data set has the following features:
	
	-	Neighborhood: 0 to 23
	-	Latitude: 49.200896849999999 to 49.31334872
	-	Longitude: -123.223955 to -123.02328940000001
	-	Year: 2003 to 2017
	- 	Month: 1 to 12
	- 	Hour: 0 to 23
	- 	Minute: 0 to 59
	- 	Day of Week: 0 to 6
	- 	Day of Month: 1 to 31

#### Code (Python files)
	- common.py
		main system file that handles packages, logging and debugging purposes
	- process.py
		preprocesses the raw data file and creates a csv with the relevant features for training, validation and testing
	- mapper.py
		since we have mapped some of the string features into integers, the mapper file preserves these features as a csv to be later used for data visualization
	- data_analysis.py
		computes statistics and generates plots for raw and processed data
	- logistic_regression.py
		runs logistic regression


##### How to Run Code
	- Download crime.csv dataset from https://www.kaggle.com/wosaku/crime-in-vancouver/downloads/crime.csv and place it in raw_data folder.
	- Fork or download zip file from this repository.
	- Run all code from the root directory with subdirectories for code and raw_data
	- First run the process.py on the original crime.csv data set in raw_data folder
	- Once process.py is completed, run mapper.py to create a map between the raw data set and features.
	- Run any other ML technique file.


##### Results for Milestone 1
	Logistic Regression
		- Avg accuracy: 0.601822160415
		- Area under roc curve: 0.70335792905
![Alt text](https://raw.githubusercontent.com/puchiha/crime_data_analysis/master/517a_crime_vancouver/plots/log_reg_ROC.png)

	Decision Tree
		- Train Accuracy ::  0.9437
		- Test Accuracy  ::  0.87695
		-- 10-fold cross-validation --
		- mean: 0.876 (std: 0.007)

##### Results for Milestone 2

	Gaussian Process Classifier
		Running on a complete dataset takes ~10-15 mins

		kernel = gp.kernels.ConstantKernel() + \
				gp.kernels.Matern(length_scale=2, nu=3/2) + \
				gp.kernels.WhiteKernel(noise_level=1)

		Accuracy: 82.30% with entire dataset

		kernel = gp.kernels.RBF(np.ones((X.shape[1], 1))) \
				 * gp.kernels.ConstantKernel() \
			     + gp.kernels.WhiteKernel()	

		Accuracy: 66.00% with 1000 data points

![Alt text](https://raw.githubusercontent.com/puchiha/crime_data_analysis/master/517a_crime_vancouver/plots/GP_results.png)

	Support Vector Machines
	        We have used an RBF kernel as well as a linear kernel
		
		RBF Kernel (Max iterations=1000, degree=2):
		- Test Accuracy  ::   0.60577
		-- 10-fold cross-validation --
		- Avg Accuracy: 0.5357
		
		Linear Kernel (Max iterations=auto)
		- Test Accuracy  :: 0.61814764783
		-- 10-fold cross-validation --
		- Avg Accuracy: 0.60192104452 

	Clustering 
		The average accuracy achieved when matching labels with the cluster centroids was 54.13 %.

##### Results for Milestone 3
	Dimensionality Reduction
	To determine the ideal number of minimaly-correlated dimensions for this data, we started by running PCA from the sklearn module in Python and examining the principal component values. The ratio of variance explained by each principle component is below
	[6.36722911e-01 1.35395400e-01 9.47514648e-02 7.15301089e-02
 	3.42552887e-02 2.04736898e-02 6.86761581e-03 2.66707205e-06
 	8.54092698e-07]

	These results suggest that much of the variance in the data can be compressed into a few dimensions.
![Alt text](https://raw.githubusercontent.com/puchiha/crime_data_analysis/master/517a_crime_vancouver/plots/svd3.png)

	Neural Network
		Training using 250 Epochs, 500 steps per epoch and 5 hidden layers with 10, 20, 13, 17, 10 neurons assigned randomly. 
	
	  1/500 [..............................] - ETA: 7s - loss: 0.1601 - acc: 0.7800
	 11/500 [..............................] - ETA: 2s - loss: 0.1727 - acc: 0.7518
	 23/500 [>.............................] - ETA: 2s - loss: 0.1708 - acc: 0.7530
	 35/500 [=>............................] - ETA: 2s - loss: 0.1725 - acc: 0.7509
	 46/500 [=>............................] - ETA: 2s - loss: 0.1702 - acc: 0.7567
	 55/500 [==>...........................] - ETA: 2s - loss: 0.1718 - acc: 0.7540
	 66/500 [==>...........................] - ETA: 2s - loss: 0.1720 - acc: 0.7527
	 77/500 [===>..........................] - ETA: 2s - loss: 0.1718 - acc: 0.7517
	 83/500 [===>..........................] - ETA: 2s - loss: 0.1721 - acc: 0.7507
	 95/500 [====>.........................] - ETA: 2s - loss: 0.1718 - acc: 0.7513
	104/500 [=====>........................] - ETA: 2s - loss: 0.1719 - acc: 0.7508
	114/500 [=====>........................] - ETA: 2s - loss: 0.1718 - acc: 0.7503
	125/500 [======>.......................] - ETA: 1s - loss: 0.1721 - acc: 0.7486
	137/500 [=======>......................] - ETA: 1s - loss: 0.1721 - acc: 0.7482
	148/500 [=======>......................] - ETA: 1s - loss: 0.1727 - acc: 0.7473
	161/500 [========>.....................] - ETA: 1s - loss: 0.1723 - acc: 0.7477
	173/500 [=========>....................] - ETA: 1s - loss: 0.1722 - acc: 0.7479
	185/500 [==========>...................] - ETA: 1s - loss: 0.1721 - acc: 0.7478
	196/500 [==========>...................] - ETA: 1s - loss: 0.1721 - acc: 0.7484
	208/500 [===========>..................] - ETA: 1s - loss: 0.1722 - acc: 0.7481
	216/500 [===========>..................] - ETA: 1s - loss: 0.1721 - acc: 0.7481
	224/500 [============>.................] - ETA: 1s - loss: 0.1722 - acc: 0.7476
	230/500 [============>.................] - ETA: 1s - loss: 0.1722 - acc: 0.7476
	239/500 [=============>................] - ETA: 1s - loss: 0.1724 - acc: 0.7469
	247/500 [=============>................] - ETA: 1s - loss: 0.1728 - acc: 0.7462
	254/500 [==============>...............] - ETA: 1s - loss: 0.1724 - acc: 0.7469
	263/500 [==============>...............] - ETA: 1s - loss: 0.1724 - acc: 0.7466
	269/500 [===============>..............] - ETA: 1s - loss: 0.1723 - acc: 0.7469
	276/500 [===============>..............] - ETA: 1s - loss: 0.1726 - acc: 0.7461
	281/500 [===============>..............] - ETA: 1s - loss: 0.1725 - acc: 0.7460
	288/500 [================>.............] - ETA: 1s - loss: 0.1727 - acc: 0.7455
	294/500 [================>.............] - ETA: 1s - loss: 0.1726 - acc: 0.7457
	301/500 [=================>............] - ETA: 1s - loss: 0.1725 - acc: 0.7459
	311/500 [=================>............] - ETA: 1s - loss: 0.1724 - acc: 0.7462
	320/500 [==================>...........] - ETA: 1s - loss: 0.1723 - acc: 0.7465
	327/500 [==================>...........] - ETA: 1s - loss: 0.1722 - acc: 0.7467
	334/500 [===================>..........] - ETA: 1s - loss: 0.1719 - acc: 0.7472
	342/500 [===================>..........] - ETA: 0s - loss: 0.1717 - acc: 0.7470
	350/500 [====================>.........] - ETA: 0s - loss: 0.1718 - acc: 0.7469
	359/500 [====================>.........] - ETA: 0s - loss: 0.1720 - acc: 0.7462
	368/500 [=====================>........] - ETA: 0s - loss: 0.1719 - acc: 0.7469
	377/500 [=====================>........] - ETA: 0s - loss: 0.1717 - acc: 0.7476
	386/500 [======================>.......] - ETA: 0s - loss: 0.1718 - acc: 0.7473
	395/500 [======================>.......] - ETA: 0s - loss: 0.1718 - acc: 0.7473
	403/500 [=======================>......] - ETA: 0s - loss: 0.1720 - acc: 0.7468
	406/500 [=======================>......] - ETA: 0s - loss: 0.1720 - acc: 0.7468
	410/500 [=======================>......] - ETA: 0s - loss: 0.1721 - acc: 0.7468
	418/500 [========================>.....] - ETA: 0s - loss: 0.1721 - acc: 0.7468
	424/500 [========================>.....] - ETA: 0s - loss: 0.1720 - acc: 0.7470
	433/500 [========================>.....] - ETA: 0s - loss: 0.1717 - acc: 0.7478
	447/500 [=========================>....] - ETA: 0s - loss: 0.1717 - acc: 0.7480
	454/500 [==========================>...] - ETA: 0s - loss: 0.1717 - acc: 0.7480
	462/500 [==========================>...] - ETA: 0s - loss: 0.1715 - acc: 0.7482
	474/500 [===========================>..] - ETA: 0s - loss: 0.1713 - acc: 0.7488
	485/500 [============================>.] - ETA: 0s - loss: 0.1713 - acc: 0.7488
	497/500 [============================>.] - ETA: 0s - loss: 0.1712 - acc: 0.7489
	500/500 [==============================] - 3s 6ms/step - loss: 0.1712 - acc: 0.7490
	__________________________________________________________________________________________________
	Layer (type)                    Output Shape         Param #     Connected to                     
	==================================================================================================
	x1 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x2 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x3 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x4 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x5 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x6 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x7 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x8 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	x9 (InputLayer)                 (None, 1)            0                                            
	__________________________________________________________________________________________________
	concatenate_1 (Concatenate)     (None, 9)            0           x1[0][0]                         
	                                                                 x2[0][0]                         
	                                                                 x3[0][0]                         
	                                                                 x4[0][0]                         
	                                                                 x5[0][0]                         
	                                                                 x6[0][0]                         
	                                                                 x7[0][0]                         
	                                                                 x8[0][0]                         
	                                                                 x9[0][0]                         
	__________________________________________________________________________________________________
	dense_1 (Dense)                 (None, 10)           100         concatenate_1[0][0]              
	__________________________________________________________________________________________________
	dense_2 (Dense)                 (None, 20)           220         dense_1[0][0]                    
	__________________________________________________________________________________________________
	dense_3 (Dense)                 (None, 13)           273         dense_2[0][0]                    
	__________________________________________________________________________________________________
	dense_4 (Dense)                 (None, 17)           238         dense_3[0][0]                    
	__________________________________________________________________________________________________
	dense_5 (Dense)                 (None, 10)           180         dense_4[0][0]                    
	__________________________________________________________________________________________________
	dense_6 (Dense)                 (None, 1)            11          dense_5[0][0]                    
	==================================================================================================
	Total params: 1,022
	Trainable params: 1,022
	Non-trainable params: 0
	__________________________________________________________________________________________________
	None

	Efficiency Test results presented in table below in Milestone 4 section.

##### Results for Milestone 4
	
	Efficiency Tests and Algorithm Comparisions
	The models were ran 10 times using 10 fold CV (except for Neural Network). The presented values are the average values of these runs. 

	Model:	mean		(std dev)	tic-toc		t-statistic 	p-value		good/bad classifier
	LR :	0.489700	(0.195078)	3.755792 s 	-7.64033	2.3653e-14		bad
	KNN:	0.516080	(0.253102)	56.94244 s 	NA 		NA 			NA
	D-Tree:	0.873000	(0.009000)	3.303621 s 	-1.743776	0.081206		good
	NB:	0.502935	(0.253945)	0.199569 s 	NA 		NA 			NA
	SVM:	0.490912	(0.297442)	221.1248 s 	4.78421	 	1.7193e-06		bad
	GP:	0.866700	(NA)		462.3874 s	-1.08796	0.28111			good
	NN:	0.749000	(NA)		770.2342 s 	254.8775	0.25079			good





