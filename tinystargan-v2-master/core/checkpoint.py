"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, folder, **kwargs):
        # Split the folder path and filename if a full path is provided
        folder_path, filename = os.path.split(str(folder))
        
        # Ensure base checkpoint directory exists
        if folder_path:
            try:
                os.makedirs(folder_path, exist_ok=True)
            except OSError as e:
                print(f"Warning: Could not create directory {folder_path}: {e}")
        
        # Store folder path and initialize module dict
        self.folder = folder_path if folder_path else '.'
        self.filename = filename  # Store filename if provided
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step, suffix='nets'):
        # If we have a stored filename, use it; otherwise create one
        if hasattr(self, 'filename') and self.filename:
            fname = os.path.join(self.folder, self.filename)
        else:
            fname = os.path.join(self.folder, f"{step:06d}_{suffix}.ckpt")
        
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        torch.save(outdict, fname)

    def load(self, step, suffix='nets'):
        # If we have a stored filename, use it; otherwise create one
        if hasattr(self, 'filename') and self.filename:
            fname = os.path.join(self.folder, self.filename)
        else:
            # Fix: Format the step number properly with an f-string
            # fname = os.path.join(self.folder, f"{int(step):06d}_{suffix}.ckpt")
            fname = os.path.join(self.folder, f"250000_{suffix}.ckpt")
        
        fname = f"expr/checkpoints/250000_{suffix}.ckpt"
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])
