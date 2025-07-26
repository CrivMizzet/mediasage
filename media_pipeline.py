#!/usr/bin/env python3
"""
Media Analysis Pipeline - Runs ollama analysis and vector embedding in sequence

Test Command
python media_pipeline.py --ollama-url "http://192.168.0.150:11434" --db-host "192.168.0.20" --db-user postgres --db-password 8g1k9ap2 --db-name media_rec --model llama3.2:latest --qdrant-host 192.168.0.20 --qdrant-port 6333 --embed-host 192.168.0.35 --embed-port 11434 --embed-model nomic-embed-text:latest --batch-size 10
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Command completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run media analysis pipeline')
    
    # Ollama analysis arguments
    parser.add_argument('--ollama-url', default='http://192.168.0.150:11434',
                       help='Ollama server URL')
    parser.add_argument('--model', default='llama3.2:latest',
                       help='Ollama model to use')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for analysis')
    
    # Database arguments
    parser.add_argument('--db-host', default='192.168.0.20',
                       help='Database host')
    parser.add_argument('--db-port', type=int, default=5432,
                       help='Database port')
    parser.add_argument('--db-user', default='postgres',
                       help='Database user')
    parser.add_argument('--db-password', required=True,
                       help='Database password')
    parser.add_argument('--db-name', default='media_rec',
                       help='Database name')
    
    # Vector embedding arguments
    parser.add_argument('--qdrant-host', default='192.168.0.20',
                       help='Qdrant host')
    parser.add_argument('--qdrant-port', type=int, default=6333,
                       help='Qdrant port')
    parser.add_argument('--embed-host', default='192.168.0.35',
                       help='Embedding server host')
    parser.add_argument('--embed-port', type=int, default=11434,
                       help='Embedding server port')
    parser.add_argument('--embed-model', default='nomic-embed-text:latest',
                       help='Embedding model')
    
    # Pipeline control
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip ollama analysis steps')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='Skip vector embedding step')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only analysis steps (skip embedding)')
    
    args = parser.parse_args()
    
    # Check if scripts exist
    scripts_dir = Path(__file__).parent
    ollama_script = scripts_dir / 'ollama_analysis.py'
    vector_script = scripts_dir / 'vector_embedding.py'
    
    if not args.skip_analysis and not ollama_script.exists():
        print(f"Error: {ollama_script} not found")
        sys.exit(1)
    
    if not args.skip_embedding and not args.analysis_only and not vector_script.exists():
        print(f"Error: {vector_script} not found")
        sys.exit(1)
    
    success_count = 0
    total_steps = 0
    
    # Run ollama analysis steps
    if not args.skip_analysis:
        analysis_types = ['content_profile', 'theme_analysis', 'mood_analysis']
        
        for analysis_type in analysis_types:
            total_steps += 1
            cmd = [
                'python', str(ollama_script),
                '--ollama-url', args.ollama_url,
                '--db-host', args.db_host,
                '--db-user', args.db_user,
                '--db-password', args.db_password,
                '--db-name', args.db_name,
                '--batch-size', str(args.batch_size),
                '--model', args.model,
                '--analysis-type', analysis_type
            ]
            
            print(f"\n--- Running {analysis_type} analysis ---")
            if run_command(cmd):
                success_count += 1
            else:
                print(f"Failed to complete {analysis_type} analysis")
                response = input("Continue with remaining steps? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
    
    # Run vector embedding step
    if not args.skip_embedding and not args.analysis_only:
        total_steps += 1
        cmd = [
            'python', str(vector_script),
            '--db-host', args.db_host,
            '--db-port', str(args.db_port),
            '--db-user', args.db_user,
            '--db-password', args.db_password,
            '--db-name', args.db_name,
            '--qdrant-host', args.qdrant_host,
            '--qdrant-port', str(args.qdrant_port),
            '--embed-host', args.embed_host,
            '--embed-port', str(args.embed_port),
            '--embed-model', args.embed_model
        ]
        
        print(f"\n--- Running vector embedding ---")
        if run_command(cmd):
            success_count += 1
        else:
            print("Failed to complete vector embedding")
            sys.exit(1)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Successfully completed {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("🎉 All steps completed successfully!")
    else:
        print("⚠️  Some steps failed - check logs above")
        sys.exit(1)

if __name__ == '__main__':
    main()
