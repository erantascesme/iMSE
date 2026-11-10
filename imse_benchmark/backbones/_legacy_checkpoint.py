from pathlib import Path
import collections, hashlib, io, pickle, tarfile, types, warnings
import torch
from torch import nn
from .. import common as f

def dense_module():
    path = Path(__file__).with_name("_densenet.py")
    source = path.read_text()
    for old, new in [('n = (depth - 4) / 3', 'n = (depth - 4) // 3'), ('n = n/2', 'n = n//2')]:
        f.require(source.count(old) == 1, 'Pinned DenseNet layer counts changed')
        source = source.replace(old, new)
    module = types.ModuleType('_pinned_odin_densenet')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

def legacy_dense_state(path, spec):
    """Read ONLY the hash-pinned Python-2 ODIN artifact through an explicit global allowlist.

    PyTorch weights_only does not support its BINSTRING opcode. This restricted
    decoder never imports a module named by the file or executes its embedded source.
    The restored legacy object is used only for state_dict; inference uses a new model.
    """
    f.require(f.hash_file(path) == spec['sha256'], 'Unrecognized legacy artifact')
    with tarfile.open(path, 'r:gz') as archive:
        member = archive.getmember(spec['member'])
        f.require(member.isfile() and member.size < 10 * 1024 ** 2, 'Unexpected tar member')
        data = archive.extractfile(member).read()
    f.require(hashlib.sha256(data).hexdigest() == spec['member_sha256'], 'Legacy member changed')
    module = dense_module()
    allowed = {('densenet', n): getattr(module, n) for n in ['DenseNet3', 'DenseBlock', 'BasicBlock', 'BottleneckBlock', 'TransitionBlock']}
    for cls in [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Sequential, nn.Linear, nn.Parameter, collections.OrderedDict, set]:
        allowed[cls.__module__, cls.__name__] = cls
    allowed.update({('__builtin__', 'set'): set, ('torch._utils', '_rebuild_tensor'): torch._utils._rebuild_tensor, ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2, ('torch._utils', '_rebuild_parameter'): torch._utils._rebuild_parameter, ('torch.nn.backends.thnn', '_get_thnn_function_backend'): lambda: None})

    class Restricted(pickle.Unpickler):

        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError('Unapproved checkpoint global: ' + module + '.' + name)
            return allowed[module, name]
    decoder = types.ModuleType('restricted_odin_pickle')
    decoder.Unpickler = Restricted
    decoder.load = lambda stream, **kw: Restricted(stream, **kw).load()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=torch.serialization.SourceChangeWarning)
        warnings.filterwarnings('ignore', message="Couldn't retrieve source code for container.*")
        legacy = torch.load(io.BytesIO(data), map_location='cpu', pickle_module=decoder, weights_only=False, encoding='latin1')
    f.require(isinstance(legacy, module.DenseNet3), 'Incorrect legacy model')
    return dict(legacy.state_dict())
