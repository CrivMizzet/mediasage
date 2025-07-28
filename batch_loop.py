#!/usr/bin/env python3
import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python loop_batch.py <number_of_times> [batch_size]")
        sys.exit(1)
    
    num_times = int(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    command = f'python batch_analysis.py --ollama-url "http://192.168.0.150:11434" --db-host 192.168.0.39 --db-user postgres --db-password 8g1k9ap2 --model mistral:latest --batch-size {batch_size}'
    
    for i in range(num_times):
        print(f"Running iteration {i+1}/{num_times}")
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()