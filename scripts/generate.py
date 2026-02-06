"""Batch Processing Script using OpenAI API"""

from batch import OpenAIClient
from tasks.context import QUERY_TYPES
import os
import sys
import time

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))

LOG_FILES = {
    "openssh": "logs/OpenSSH_2k.log", 
    "linux": "logs/Linux_2k.log",
    "apache": "logs/Apache_2k.log",
    "hadoop": "logs/Hadoop_2k.log",
    "mac": "logs/Mac_2k.log",
    "hdfs": "logs/HDFS_2k.log",
    "hpc": "logs/HPC_2k.log",
    "openstack": "logs/OpenStack_2k.log",
    "spark": "logs/Spark_2k.log",
    "bgl": "logs/BGL_2k.log", 
    "health": "logs/HealthApp_2k.log", 
    "prox": "logs/Proxifier_2k.log", 
    "thunderbird": "logs/Thunderbird_2k.log", 
    "zookeeper": "logs/ZooKeeper_2k.log"
}

#Poll batch status until done, then download results
def wait_and_retrieve(client, batch_id, save_path):
    print("\nWaiting for batch to complete...")
    while True:
        status = client.check_batch_status(batch_id)
        print(f"[{time.strftime('%H:%M:%S')}] Batch status: {status}")

        if status == "completed":
            output_file_id = client.get_output_file_id(batch_id)
            client.retrieve_batch_results(output_file_id, save_path)
            return True
        elif status in ["failed", "expired", "cancelled"]:
            print(f"✗ Batch {status}!")
            return False
        
        time.sleep(60)


def generate(): 
    client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), encoding_model="gpt2")

    # Batch 1: Summary queries (~1685 requests)
    print("\n" + "="*60)
    print("BATCH 1: SUMMARY QUERIES")
    print("="*60)
    batch_file = client.create_batch_summary_input(LOG_FILES, "data/batch_input_summary.jsonl")
    batch_id = client.submit_batch(batch_file)
    with open("batch_id_summary.txt", "w") as f:
        f.write(batch_id)
    wait_and_retrieve(client, batch_id, "data/batch_output_summary.jsonl")

    # Batch 2+: Varied queries — one batch per query type to stay within token limits
    for query_name in QUERY_TYPES:
        print("\n" + "="*60)
        print(f"BATCH: {query_name.upper()} QUERIES")
        print("="*60)
        batch_file = client.create_varied_batch_input(
            LOG_FILES, f"data/batch_input_{query_name}.jsonl", query_types=[query_name]
        )
        batch_id = client.submit_batch(batch_file)
        with open(f"batch_id_{query_name}.txt", "w") as f:
            f.write(batch_id)
        wait_and_retrieve(client, batch_id, f"data/batch_output_{query_name}.jsonl")

    print("\n✓ DONE!")


if __name__ == "__main__":
    generate()


