#!/usr/bin/env python3
"""
Script to run ollama_analysis.py with multiple analysis types sequentially.

python batch_analysis.py --ollama-url "http://192.168.0.150:11434" --db-host 192.168.0.39 --db-user postgres --db-password 8g1k9ap2 --model mistral:latest --batch-size 20
"""

import subprocess
import sys
import argparse

def main():
    # Parse arguments to extract base parameters
    parser = argparse.ArgumentParser(description='Run multiple ollama analysis types')
    parser.add_argument('--ollama-url', required=True, help='Ollama server URL')
    parser.add_argument('--db-host', required=True, help='Database host')
    parser.add_argument('--db-user', required=True, help='Database user')
    parser.add_argument('--db-password', required=True, help='Database password')
    parser.add_argument('--batch-size', required=True, help='Batch size')
    parser.add_argument('--model', required=True, help='Model name')
    
    args = parser.parse_args()
    
    # Analysis types to run in order
    analysis_types = [
        'content_profile',
        'mood_analysis',
        'theme_analysis',
        'similarity_analysis',
        'recommendation_profile'
        
    ]
    
    # Base command arguments
    base_cmd = [
        'python', 'ollama_analysis.py',
        '--ollama-url', args.ollama_url,
        '--db-host', args.db_host,
        '--db-user', args.db_user,
        '--db-password', args.db_password,
        '--batch-size', args.batch_size,
        '--model', args.model
    ]
    
    # Run each analysis type
    for analysis_type in analysis_types:
        cmd = base_cmd + ['--analysis-type', analysis_type]
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Completed {analysis_type}")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {analysis_type}: {e}")
            print(f"stderr: {e.stderr}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n✗ Interrupted during {analysis_type}")
            sys.exit(1)
    
    print("All analysis types completed successfully!")

if __name__ == '__main__':
    main()