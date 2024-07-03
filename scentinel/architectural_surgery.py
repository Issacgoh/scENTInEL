import scvi
import anndata
import torch

def update_scvi_model_with_new_data_transfer_learning(old_adata, new_adata, trained_model, max_epochs=20, weight_decay=0.0):
    """
    Update a trained scVI model with new single-cell data using transfer learning and get a new latent representation.

    Parameters:
    old_adata (anndata.AnnData): AnnData object containing the old single-cell data.
    new_adata (anndata.AnnData): AnnData object containing the new single-cell data.
    trained_model (scvi.model.SCVI): A trained scVI model on the old data.
    max_epochs (int): Number of epochs to train on the combined dataset.
    weight_decay (float): Weight decay parameter for training.

    Returns:
    new_adata_with_latent (anndata.AnnData): Combined AnnData object with updated latent representations.
    """
    
    # Ensure the new data has the same variables as the old data
    new_adata = new_adata[:, old_adata.var_names].copy()
    
    # Concatenate old and new data
    combined_adata = old_adata.concatenate(new_adata, batch_key="batch")
    
    # Register the new combined AnnData with the trained model setup
    scvi.data.setup_anndata(combined_adata, batch_key="batch")
    
    # Transfer the trained model's setup to the combined AnnData
    trained_model._register_anndata(combined_adata)

    # Freeze layers if necessary to retain old knowledge (optional step)
    for param in trained_model.module.parameters():
        param.requires_grad = True

    # Use transfer learning: continue training the existing model on the combined dataset
    trained_model.train(max_epochs=max_epochs, plan_kwargs={'weight_decay': weight_decay})
    
    # Get the latent representation for the combined dataset
    combined_adata.obsm["X_scVI"] = trained_model.get_latent_representation(combined_adata)
    
    return combined_adata

# Example usage:
# new_adata_with_latent = update_scvi_model_with_new_data_transfer_learning(old_adata, new_adata, trained_model)
