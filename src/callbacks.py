"""
This module defines a custom PyTorch Lightning callback for performing rollouts and logging results during training and validation.
"""
import os
import lightning.pytorch as pl
from pathlib import Path
import wandb
import shutil
from src.utils.plots import plot_2D, plot_3D, plot_2D_image, plot_image3D, plotError, video_plot_3D
from src.evaluate import roll_out, compute_error, print_error

# Define the RolloutCallback class
class RolloutCallback(pl.Callback):
    def __init__(self, dataloader, **kwargs):
        # Initialize the callback with a dataloader and additional arguments
        super().__init__(**kwargs)
        self.dataloader = dataloader

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # Reset rollout-related attributes at the start of each validation epoch
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

        # Remove the 'videos' folder if it exists
        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            shutil.rmtree(folder_path)  # Remove the folder and its contents

    def on_validation_epoch_end(self, trainer, pl_module):
        # Perform rollout and logging at the end of validation epochs based on frequency
        if trainer.current_epoch > 0 and trainer.current_epoch % pl_module.rollout_freq == 0:
            try:
                # Perform rollout to get predictions and ground truth
                z_net, z_gt, t = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.dtset_type)

                # Save the rollout as a GIF
                save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}.gif')
                if pl_module.data_dim == 2:
                    plot_2D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir, var=5)  # Plot 2D rollout
                else:
                    video_plot_3D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir)  # Plot 3D rollout

                # Log the rollout video to WandB
                trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})
            except:
                print('The rollout has failed')  # Handle rollout failure

    def on_train_end(self, trainer, pl_module):
        # Perform final rollout and save metrics at the end of training
        z_net, z_gt, q_0 = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.dtset_type)
        filePath = os.path.join(pl_module.save_folder, 'metrics.txt')  # Path to save metrics
        save_dir = os.path.join(pl_module.save_folder, f'final_{trainer.current_epoch}.gif')  # Path to save final GIF

        # Compute and save error metrics
        with open(filePath, 'w') as f:
            error, L2_list = compute_error(z_net, z_gt, pl_module.state_variables)  # Compute errors
            lines = print_error(error)  # Format error messages
            f.write('\n'.join(lines))  # Write errors to file
            print("[Test Evaluation Finished]\n")
            f.close()

        # Plot error metrics and save visualizations
        plotError(z_gt, z_net, L2_list, pl_module.state_variables, pl_module.data_dim, pl_module.save_folder)
        if pl_module.data_dim == 2:
            plot_2D(z_net, z_gt, save_dir=save_dir, var=5)  # Plot 2D final rollout
            plot_2D_image(z_net, z_gt, -1, var=5, output_dir=pl_module.save_folder)  # Save 2D images
        else:
            plot_3D(z_net, z_gt, save_dir=save_dir, var=5)  # Plot 3D final rollout
            data = [sample for sample in self.dataloader]  # Get data samples
            plot_image3D(z_net, z_gt, pl_module.save_folder, var=5, step=-1, n=data[0].n)  # Save 3D images



