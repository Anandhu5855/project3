import subprocess
import time
import os

def start_spark_master():
    master_cmd = r'C:\spark\bin\spark-class.cmd org.apache.spark.deploy.master.Master'
    subprocess.Popen(master_cmd, shell=True)
    print("Spark Master started.")
    time.sleep(5)  # Give it a few seconds to initialize

def start_spark_worker(master_url="spark://172.18.111.237:7077"):
    worker_cmd = fr'C:\spark\bin\spark-class.cmd org.apache.spark.deploy.worker.Worker {master_url}'
    subprocess.Popen(worker_cmd, shell=True)
    print("Spark Worker started.")
    time.sleep(5)

def start_spark_cluster():
    start_spark_master()
    start_spark_worker()

# Example usage
if __name__ == "__main__":
    start_spark_cluster()
