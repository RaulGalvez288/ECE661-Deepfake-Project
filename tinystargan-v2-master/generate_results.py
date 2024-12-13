import os
import subprocess
import fileinput
import shutil

def format_iteration(iteration):
    """Format iteration number according to checkpoint naming convention"""
    iter_num = iteration // 1000
    print(iter_num)
    if iteration < 100000:
        return f"0{iter_num}000"  # e.g., "060000" for 60k
    else:
        return f"{iter_num}000"   # e.g., "100000" for 100k

def modify_checkpoint_file(iteration):
    """Modify the checkpoint.py file with the current iteration"""
    iteration_str = format_iteration(iteration)
    checkpoint_path = "core/checkpoint.py"
    
    # Create backup if it doesn't exist
    if not os.path.exists(checkpoint_path + ".backup"):
        shutil.copy2(checkpoint_path, checkpoint_path + ".backup")
    else:
        # Restore from backup to ensure clean state
        shutil.copy2(checkpoint_path + ".backup", checkpoint_path)
    
    # Read the file and modify the lines
    with fileinput.FileInput(checkpoint_path, inplace=True) as file:
        print("THIS IS THE ITERATION", iteration_str)
        for line_number, line in enumerate(file, 1):
            if line_number == 55:

                print('            fname = os.path.join(self.folder, f"{}_{{suffix}}.ckpt")'.format(iteration_str))
            elif line_number == 57:
                print('        fname = f"expr/checkpoints/{}_{{suffix}}.ckpt"'.format(iteration_str))
            else:
                print(line, end='')

def run_sample_command(iteration):
    """Run the sample command for a specific iteration"""
    result_dir = f"expr/results_{iteration//1000}_64"
    
    # Skip if results directory already exists
    if os.path.exists(result_dir):
        print(f"Skipping iteration {iteration} - results directory already exists")
        return
    
    # Check if checkpoint exists before proceeding
    checkpoint_file = f"expr/checkpoints/{format_iteration(iteration)}_nets_ema.ckpt"
    if not os.path.exists(checkpoint_file):
        print(f"Skipping iteration {iteration} - checkpoint file {checkpoint_file} does not exist")
        return
    
    # Modify checkpoint.py
    modify_checkpoint_file(iteration)
    
    # Construct and run the command
    command = [
        'python', 'main.py',
        '--mode', 'sample',
        '--num_domains', '2',
        '--w_hpf', '1',
        '--alpha', '64',
        '--efficient', '1',
        '--resume_iter', str(iteration),
        '--checkpoint_dir', 'expr/checkpoints/',
        '--val_batch_size', '64',
        '--latent_sample_per_domain', '500',
        '--filename', 'tiny_celeb_male.jpg',
        '--src_dir', 'assets/representative/celeb_male/src',
        '--ref_dir', 'assets/representative/celeb_male/ref',
        '--result_dir', result_dir
    ]
    
    try:
        print(f"\nProcessing iteration {iteration}...")
        subprocess.run(command, check=True)
        print(f"Successfully processed iteration {iteration}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing iteration {iteration}: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        if os.path.exists("core/checkpoint.py.backup"):
            shutil.copy2("core/checkpoint.py.backup", "core/checkpoint.py")
            os.remove("core/checkpoint.py.backup")
            print("Restored original checkpoint.py")
        raise

def main():
    try:
        # Create range of iterations from 20k to 300k, stepping by 20k
        iterations = range(20000, 300001, 20000)
        
        # Process each iteration
        for iteration in iterations:
            run_sample_command(iteration)
            print(f"Iteration {iteration} completed")
        
    finally:
        # Restore original checkpoint.py from backup
        if os.path.exists("core/checkpoint.py.backup"):
            shutil.copy2("core/checkpoint.py.backup", "core/checkpoint.py")
            os.remove("core/checkpoint.py.backup")
            print("Restored original checkpoint.py")

if __name__ == "__main__":
    main()