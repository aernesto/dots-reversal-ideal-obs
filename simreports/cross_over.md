# Simulation report

This report is an answer to [this simulation plan](https://github.com/aernesto/dots-reversal-ideal-obs/blob/abda640a9de98ea32031344dc8a07ef55848933d/simgoals/cross_over.txt).  

The first part of the plan, dealing with generating a database of change point locations, was realized with [this script](https://github.com/aernesto/dots-reversal-ideal-obs/blob/55c7f04929eb3e32d80a3a2fbada0b415975e8e6/simscripts/gen_trials.py). Running it on a laptop took 12 min 13 sec.   
The resulting database is available [here](https://www.dropbox.com/s/bvxgawscdojeb1d/true_2.db?dl=0).

A quick inspection of the database makes sense. The following is pasted from my Unix terminal:
```
$ sqlite3 true_2.db 
SQLite version 3.13.0 2016-05-18 10:57:30
Enter ".help" for usage hints.
sqlite> select COUNT(*) from crossover;
80000
sqlite> select distinct h from crossover;
0.1
0.5
1.0
1.5
2.0
sqlite> select distinct snr from crossover;
0.5
1.0
1.5
2.0
sqlite> select count(*) from crossover group by numb;
10000
10000
10000
10000
10000
10000
10000                                                                                                     
10000                                                                                               
sqlite> select count(*) from crossover where snr = 1 group by numb;                                                      
2500                                                                                                                     
2500                                                                                                                  
2500                                                                                 
2500                                                                                                             
2500                                                                                                                 
2500                                                                        
2500                                                                                               
2500
```
