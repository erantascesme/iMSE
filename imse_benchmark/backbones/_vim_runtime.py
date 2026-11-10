from pathlib import Path
import importlib
import subprocess
import sys
import torch
from ..common import require

VIM_COMMIT = 'dd0358ad1e42701f22afbefa0717cc8825cf9f45'
VIM_FACTORY = 'vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2'


def _compat(VIM_REPO):
    VIM_SOURCE_DIR = VIM_REPO / 'vim'
    VIM_MODELS_FILE = VIM_SOURCE_DIR / 'models_mamba.py'
    assert VIM_MODELS_FILE.exists()
    subprocess.run(['git', '-C', str(VIM_REPO), 'checkout', '--', 'vim/models_mamba.py'], check=True)
    print('✅ Restored official Vim models_mamba.py from Git.')
    vim_source = VIM_MODELS_FILE.read_text(encoding='utf-8')
    if 'if_devide_out=True' in vim_source:
        vim_source = vim_source.replace('if_devide_out=True', 'if_divide_out=True')
        VIM_MODELS_FILE.write_text(vim_source, encoding='utf-8')
        print('✅ Patched Vim keyword: if_devide_out -> if_divide_out')
    else:
        print('ℹ️ Vim keyword typo not present; official source needs no keyword patch.')
    import py_compile
    py_compile.compile(str(VIM_MODELS_FILE), doraise=True)
    print('✅ Official Vim models_mamba.py syntax validated.')
    MAMBA_SOURCE_ROOT = VIM_REPO / 'mamba-1p1p1'
    CAUSAL_SOURCE_ROOT = VIM_REPO / 'causal-conv1d'
    assert (MAMBA_SOURCE_ROOT / 'mamba_ssm').is_dir(), 'Official vendored mamba_ssm source is missing.'
    for source_root in [MAMBA_SOURCE_ROOT, CAUSAL_SOURCE_ROOT]:
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    import importlib
    import importlib.util

    def expose_compiled_extension(module_name, search_roots):
        """
        Ensure a top-level compiled CUDA extension produced during
        Cell 1 is importable. Editable builds can leave the .so in
        the source/build tree without exposing that directory under
        Python 3.13.
        """
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
            print(f'✅ {module_name} importable:', getattr(module, '__file__', '<built-in>'))
            return module
        except Exception as first_error:
            candidates = []
            for root in search_roots:
                if not root.exists():
                    continue
                candidates.extend(root.rglob(f'{module_name}*.so'))
            py_tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
            candidates = sorted(candidates, key=lambda p: (py_tag not in p.name, len(str(p))))
            last_error = first_error
            for candidate in candidates:
                parent = str(candidate.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                importlib.invalidate_caches()
                sys.modules.pop(module_name, None)
                try:
                    module = importlib.import_module(module_name)
                    print(f'✅ {module_name} exposed from:', candidate)
                    return module
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f'Cell 1 did not leave an importable {module_name} CUDA extension. Found {len(candidates)} candidate .so files. First import error: {first_error!r}. Last retry error: {last_error!r}.') from last_error
    selective_scan_cuda = expose_compiled_extension('selective_scan_cuda', [MAMBA_SOURCE_ROOT, Path('/usr/local/lib/python3.13/dist-packages')])
    causal_conv1d_cuda = expose_compiled_extension('causal_conv1d_cuda', [CAUSAL_SOURCE_ROOT, Path('/usr/local/lib/python3.13/dist-packages')])
    MAMBA_GENERATION_FILE = MAMBA_SOURCE_ROOT / 'mamba_ssm' / 'utils' / 'generation.py'
    assert MAMBA_GENERATION_FILE.exists()
    generation_source = subprocess.check_output(['git', '-C', str(VIM_REPO), 'show', 'HEAD:mamba-1p1p1/mamba_ssm/utils/generation.py'], text=True, encoding='utf-8')
    legacy_import = 'from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer'
    compat_import = 'try:\n    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer\nexcept ImportError:\n    try:\n        from transformers.generation import TextStreamer\n    except ImportError:\n        from transformers.generation.streamers import TextStreamer\n    GreedySearchDecoderOnlyOutput = namedtuple(\n        "GreedySearchDecoderOnlyOutput", ["sequences", "scores"]\n    )\n    SampleDecoderOnlyOutput = namedtuple(\n        "SampleDecoderOnlyOutput", ["sequences", "scores"]\n    )\n'
    if generation_source.count(legacy_import) != 1:
        raise RuntimeError('Unexpected official Mamba generation.py import layout. No replacement was written. Check the Vim Git revision.')
    generation_source = generation_source.replace(legacy_import, compat_import, 1)
    compile(generation_source, str(MAMBA_GENERATION_FILE), 'exec')
    MAMBA_GENERATION_FILE.write_text(generation_source, encoding='utf-8')
    py_compile.compile(str(MAMBA_GENERATION_FILE), doraise=True)
    print('✅ Restored and patched vendored Mamba generation.py; syntax validated.')
    for module_name in list(sys.modules.keys()):
        if module_name == 'mamba_ssm' or module_name.startswith('mamba_ssm.'):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    mamba_spec = importlib.util.find_spec('mamba_ssm')
    assert mamba_spec is not None, f'mamba_ssm is still not visible after adding {MAMBA_SOURCE_ROOT} to sys.path.'
    import mamba_ssm
    print('✅ mamba_ssm importable from:', mamba_ssm.__file__)
    mamba_file = Path(mamba_ssm.__file__).resolve()
    assert MAMBA_SOURCE_ROOT.resolve() in mamba_file.parents, f'Unexpected mamba_ssm package is being used: {mamba_file}. Expected the official Vim vendored copy under {MAMBA_SOURCE_ROOT}.'
    print('✅ Using official Vim vendored mamba_ssm.')
    if str(VIM_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(VIM_SOURCE_DIR))
    import importlib
    models_mamba = importlib.import_module('models_mamba')
    VIM_FACTORY_NAME = 'vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2'
    assert hasattr(models_mamba, VIM_FACTORY_NAME), 'Official Vim base factory was not found. Do not substitute another Vim architecture.'
    return getattr(models_mamba, VIM_FACTORY)

def prepare_vim_source(root):
    root=Path(root)
    if not root.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/hustvl/Vim.git',str(root)],check=True)
        current=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if current!=VIM_COMMIT:
            subprocess.run(['git','-C',str(root),'fetch','--depth','1','origin',VIM_COMMIT],check=True)
            subprocess.run(['git','-C',str(root),'checkout','--detach',VIM_COMMIT],check=True)
    current=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    require(current==VIM_COMMIT,'Existing Vim source is at a different commit; choose a separate VIM_SOURCE_ROOT')
    for part in ['mamba-1p1p1','causal-conv1d']:
        path=str(root/part)
        if path not in sys.path:sys.path.insert(0,path)
    needs_build=False
    for name in ['selective_scan_cuda','causal_conv1d_cuda']:
        try:importlib.import_module(name)
        except (ImportError,OSError):needs_build=True
    if needs_build:
        require(torch.cuda.is_available(),'Vim extension preparation requires a CUDA runtime')
        print('Building the pinned Vim extensions once for this runtime; existing feature banks are untouched.')
        for part in ['causal-conv1d','mamba-1p1p1']:
            subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(root/part)],check=True)
    return _compat(root)
