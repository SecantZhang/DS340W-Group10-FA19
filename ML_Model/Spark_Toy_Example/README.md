# Instructions for Running Spark/Hadoop on Bridges System. 

## Step 1 - System-Wise Preparation: 
In order to use the hadoop system which includes the HDFS and Hadoop, you should request a interact session on Bridges. 
```
interact -N <number of nodes>
```
After successfully allocated the interactive session, you need to load the hadoop module into the system. 
```
module load hadoop
start-hadoop.sh
```