# Simulation report

This report is an answer to [this simulation plan](https://github.com/aernesto/dots-reversal-ideal-obs/blob/abda640a9de98ea32031344dc8a07ef55848933d/simgoals/cross_over.txt).  

## First failure
### Script and database
The first part of the plan, dealing with generating a database of change point locations, was realized with [this script](https://github.com/aernesto/dots-reversal-ideal-obs/blob/abda640a9de98ea32031344dc8a07ef55848933d/simscripts/gen_trials.py). Running it on a laptop took 1 h 15 min 38 sec.   
The resulting database with is available [here](https://www.dropbox.com/s/k444qi8ws9qaln3/true_1.db?dl=0).

### The problem
According to the console output during the run, the outer while loop was only executed 11 times, when it should have been executed 20 times. **Why did this happen ?**  
Also, investigating the resulting SQLite database file, it appears that the database only contains 4,000 rows, when the script is meant to write 80,000 rows. Inspecting the columns 'h' and 'snr', it appears that only one value of 'h' (0.1) and one value of 'snr' (0.5) were written in the database. **Why?**
