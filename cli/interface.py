
import argparse
import sys
import time
from pathlib import Path

from cli.processor import LogProcessor
from logpt.inference import LogAnalyzer, QUERY_PROMPTS

BANNER = """
══════════════════════════════════════════
  LoGPT — Log Analysis
  GPT-2 124M · Fine-tuned · On-Device
══════════════════════════════════════════
"""

#Function to find the best model checkpoint 
def find_checkpoint(): 
    candidates = [
        Path("logs/best_model.pth"),
        Path("../logs/best_model.pth"),
        Path(__file__).parent.parent / "logs" / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No checkpoint found in expected locations.")

#Will read log content from a file or stdin
def read_log(file_path): 

    if file_path: 
        path = Path(file_path)
        if not path.exists():
            print(f"Log file not found: {file_path}")
            sys.exit(1)
        return path.read_text(encoding='utf-8', errors='replace'), path.name
    
    # this will check if there's data being piped in from stdin
    if not sys.stdin.isatty():
        content = sys.stdin.read()
        if not content.strip():
            print("Error: empty input from stdin.", file=sys.stderr)
            sys.exit(1)
        return content, "stdin"
    
    print("Error: provide --file or pipe input via stdin.", file=sys.stderr)
    sys.exit(1)

def main():
    #OPTIONS AND ARGUMENTS
    parser = argparse.ArgumentParser(
        prog="logpt",
        description="Analyze log files locally with a fine-tuned GPT-2 model.",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to the log file (or pipe via stdin).",
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True,
        choices=["summary", "root_cause", "action_items"],
        help="Type of analysis to perform.",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to model checkpoint (default: auto-detect).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate per chunk (default: 256).",
    )

    args = parser.parse_args()

    #Read the input 
    log_content, source = read_log(args.file)
    total_lines = log_content.count('\n') + 1
    print(f"  Source: {source} ({total_lines:,} lines)", file=sys.stderr)
    print(f"  Query:  {args.query}", file=sys.stderr)

    #Load the model 
    model_path = args.model or find_checkpoint()
    analyzer = LogAnalyzer(model_path)
    
    #Chunk the log 
    processor = LogProcessor(max_seq_len=analyzer.max_seq_len, response_reserve=args.max_tokens)
    chunks = processor.chunk(log_content)
    total_chunks = len(chunks)

    print(f'  Total Chunks: {total_chunks}', file=sys.stderr)

    #Generate the response per chunk with user paging 
    for i, chunk in enumerate(chunks): 
        chunk_lines = chunk.count('\n') + 1
        header = f"=== Chunk {i+1}/{total_chunks} ({chunk_lines} lines) ==="
        header += "-" * max(0, 50 - len(header))
        print(header)
        print()

        t0 = time.time()
        response = analyzer.generate(
            log_content=chunk,
            query_type=args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        elasped = time.time() - t0
        print(response)
        print(f"\n[Chunk {i+1} generated in {elasped:.1f} seconds]")

        #if there are more chunks, ask the user if they want to continue
        if i < total_chunks - 1:
            print("─" * 50)
            answer  = input(f"\nContinue to chunk {i + 2}/{total_chunks}? [Y/n] ").strip().lower()
            if answer and answer[0] != 'y':
                print("Exiting.")
                break
        else:
            print(f"\n  Done. Analyzed {total_chunks} chunk(s) from {source}.") 


if __name__ == "__main__":
    main()
     

    