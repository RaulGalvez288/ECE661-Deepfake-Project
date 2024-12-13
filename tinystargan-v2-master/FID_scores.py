import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random

def extract_fid_score(output):
    match = re.search(r"FID:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None

def copy_and_align_images(checkpoint, num_images=5000):
    # Create directories
    fake_dir = f"./expr/results_{checkpoint//1000}_64"
    test_fake_dir = f"./test{checkpoint//1000}_fake_64"
    test_fake_aligned_dir = f"./test{checkpoint//1000}_fake_aligned_64"
    
    # Check if aligned directory already has enough images
    if os.path.exists(test_fake_aligned_dir):
        aligned_images = os.listdir(test_fake_aligned_dir)
        if len(aligned_images) >= num_images:
            print(f"Already have {len(aligned_images)} aligned images in {test_fake_aligned_dir}, skipping...")
            return test_fake_aligned_dir
    
    # Create test directory if it doesn't exist
    os.makedirs(test_fake_dir, exist_ok=True)
    os.makedirs(test_fake_aligned_dir, exist_ok=True)
    
    # Clear existing files in test directory
    for file in os.listdir(test_fake_dir):
        os.remove(os.path.join(test_fake_dir, file))
    
    # Copy random images
    all_images = os.listdir(fake_dir)
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    for img in selected_images:
        shutil.copy2(os.path.join(fake_dir, img), test_fake_dir)
    
    print(f"Copied {len(selected_images)} images to {test_fake_dir}")
    
    # Align images
    try:
        align_command = [
            'python', 'main.py',
            '--mode', 'align',
            '--inp_dir', test_fake_dir,
            '--out_dir', test_fake_aligned_dir,
            '--wing_path', 'expr/checkpoints/wing.ckpt',
            '--lm_path', 'expr/checkpoints/celeba_lm_mean.npz'
        ]
        subprocess.run(align_command, check=True)
        print(f"Aligned images saved in {test_fake_aligned_dir}")
        return test_fake_aligned_dir
    except subprocess.CalledProcessError as e:
        print(f"Error during alignment: {e}")
        return None

def calculate_and_plot_fid():
    # Initialize arrays for plotting
    checkpoints = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]) * 1000
    fid_scores = []
    
    # Ensure real directory exists and is aligned
    real_aligned_dir = "./test_real2_aligned"
    
    # Align real images if not already done
    # Calculate FID for each checkpoint
    for checkpoint in checkpoints:
        print(f"\nProcessing checkpoint {checkpoint}...")
        
        # Copy and align fake images
        fake_aligned_dir = copy_and_align_images(checkpoint)
        if not fake_aligned_dir:
            print(f"Skipping FID calculation for checkpoint {checkpoint}")
            fid_scores.append(None)
            continue
        
        # Calculate FID
        try:
            result = subprocess.run(
                ['python', '-m', 'pytorch_fid', '--dims', '2048', 
                 fake_aligned_dir, real_aligned_dir],
                capture_output=True,
                text=True,
                check=True
            )
            
            fid_score = extract_fid_score(result.stdout)
            if fid_score is not None:
                fid_scores.append(fid_score)
                print(f"Checkpoint {checkpoint}: FID = {fid_score}")
            else:
                print(f"Failed to extract FID score for checkpoint {checkpoint}")
                fid_scores.append(None)
                
        except subprocess.CalledProcessError as e:
            print(f"Error calculating FID for checkpoint {checkpoint}: {e}")
            fid_scores.append(None)
    
    # Convert to numpy array and remove any None values
    valid_scores = [(ckpt, score) for ckpt, score in zip(checkpoints, fid_scores) if score is not None]
    if not valid_scores:
        print("No valid FID scores calculated")
        return
        
    checkpoints_clean, fid_scores_clean = zip(*valid_scores)
    checkpoints_clean = np.array(checkpoints_clean)
    fid_scores_clean = np.array(fid_scores_clean)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints_clean, fid_scores_clean, marker='o')
    plt.xlabel('Checkpoint Iteration')
    plt.ylabel('FID Score')
    plt.title('FID Score vs Training Iteration')
    plt.grid(True)
    
    
    # Save the plot
    plt.savefig('fid_scores_plot.png')
    print("\nPlot saved as 'fid_scores_plot.png'")
    
    # Save the raw data
    with open('fid_scores.txt', 'w') as f:
        f.write("Checkpoint,FID Score\n")
        for checkpoint, score in zip(checkpoints_clean, fid_scores_clean):
            f.write(f"{checkpoint},{score}\n")
    print("Raw scores saved in 'fid_scores.txt'")

if __name__ == "__main__":
    calculate_and_plot_fid()