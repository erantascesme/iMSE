import torch
from ..common import require, atomic_json

def _load_swin_state_strict(model, state, audit_path):
    import timm
    from timm.models.swin_transformer_v2 import checkpoint_filter_fn
    require(timm.__version__ == '1.0.11', 'This compatibility patch requires timm 1.0.11')
    filtered = checkpoint_filter_fn(state, model)
    removed = sorted(set(state) - set(filtered))
    allowed = {'attn_mask', 'relative_position_index', 'relative_coords_table'}
    require(all((k.rsplit('.', 1)[-1] in allowed for k in removed)), 'Swin filter attempted to remove a learned parameter')
    require(set(filtered).issubset(state), 'Unexpected Swin checkpoint key remapping')
    require(all((filtered[k] is state[k] for k in filtered)), 'Unexpected resizing or modification of Swin checkpoint tensors')
    buffers = dict(model.named_buffers())
    for key in removed:
        require(key in buffers, 'Removed key is not a regenerated model buffer: ' + key)
        require(torch.equal(state[key].cpu(), buffers[key].cpu()), 'Saved/regenerated Swin buffer differs at the selected resolution: ' + key)
    result = model.load_state_dict(filtered, strict=True)
    atomic_json(audit_path, dict(removed_regenerated_buffers=removed, regenerated_buffers_equal=True, learned_tensors_modified=False, strict=True, timm_version=timm.__version__))
    print('Swin checkpoint loaded strictly; verified regenerated buffers:', removed)
    return result
