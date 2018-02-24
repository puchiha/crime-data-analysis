Goal: Identify and Run Analysis on crime data relating to collisions ...
Objective: 

Git Structire:
	- data
		// this is the root folder, use this folder to run codes (i.e. $ python code/process.py)

		- code (uploaded to github)
		- raw_data
		- results
		- plots


- Data_set 
	- crime.csv ([530652 rows x 12 columns])
	
	The different columns in the dataset are:  ['TYPE', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD', 'X', 'Y', 'Latitude', 'Longitude', 'DATE', 'DAY_OF_WEEK', 'CLASSIFICATION']

	9 types of crime but we only focus on collision data. (if 'collision', +1, otherwise -1)
	24 different neighborhoods.

	date/time: save date for visualization purposes only. change date ...

	- crime_processed.csv

	- process.py
	// 
