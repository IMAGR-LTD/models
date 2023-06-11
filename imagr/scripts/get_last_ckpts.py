import tensorflow as tf
import argparse

def run(checkpoint_dir: str, output_filename: str):
    print(f'Searching for checkpoints in {checkpoint_dir}')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print(f'Found {latest_checkpoint}')
    
    with open(f"{checkpoint_dir}/{output_filename}", "w") as fd:
        fd.write(latest_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find the latest tensorflow checkpoints'
    )

    parser.add_argument('checkpoint_dir') 
    parser.add_argument('output_filename')
    args = parser.parse_args()

    run(args.checkpoint_dir, args.output_filename)
