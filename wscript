#!/usr/bin/env python
from waflib.extras.test_base import summary
import copy
import site
import os
import re
import sys
from waflib.extras.symwaf2ic import get_toplevel_path

def depends(dep):
    dep('hxcomm')
    dep('haldls')
    dep('grenade')


def options(opt):
    opt.load('compiler_cxx')
    opt.load('test_base')
    opt.load('python')
    opt.load('pytest')
    opt.load('pylint')
    opt.load('pycodestyle')
    opt.load('doxygen')


def configure(cfg):
    cfg.load('compiler_cxx')
    cfg.load('test_base')
    cfg.load('python')
    cfg.check_python_version()
    cfg.check_python_headers()
    cfg.load('pytest')
    cfg.load('doxygen')

    cfg.check(
        compiler='cxx',
        features='cxx pyext',
        uselib_store='PYBIND11HXTORCH',
        mandatory=True,
        header_name='pybind11/pybind11.h',
    )

    site_packages = site.getsitepackages()
    assert isinstance(site_packages, list) and len(site_packages) == 1
    includes_torch = [os.path.join(x, 'torch/include') for x in site_packages]
    includes_torch_csrc_api = [os.path.join(x, 'torch/include/torch/csrc/api/include') for x in site_packages]
    libpath_torch = [os.path.join(x, 'torch/lib') for x in site_packages]
    libnames = []
    # if torch isn't available via site-packages, try sys.path/PYTHONPATH
    if not os.path.exists(libpath_torch[0]):
        libpath_torch = [os.path.join(x, 'torch/lib') for x in sys.path if 'torch' in x]
        if len(libpath_torch) == 0:
            cfg.fatal('PyTorch library directory not found')
        elif len(libpath_torch) > 1:
            cfg.fatal('More than one location for PyTorch libraries found: {}'.format(', '.join(libpath_torch)))
        libpath_torch = libpath_torch[0]
    for fn in os.listdir(libpath_torch[0]):
        res = re.match('^lib(.+)\.so$', fn)
        libnames.append(res.group(1))

    libnames_cpp = copy.copy(libnames)
    libnames_cpp.remove('torch_python')
    cfg.check_cxx(fragment ='''
                    #include <torch/torch.h>
                    #include <torch/csrc/jit/runtime/custom_operator.h>
                    int main() { return 0; }''',
                  lib = libnames_cpp,
                  libpath = libpath_torch,
                  cxxflags = map(lambda x: '-isystem' + x, (includes_torch_csrc_api + includes_torch)),
                  uselib_store="TORCH_CPP")
    # manually add the torch includes as system includes
    cfg.env['CXXFLAGS_TORCH_CPP'] += map(lambda x: '-isystem' + x, (includes_torch_csrc_api + includes_torch))

    cfg.check_cxx(fragment ='''
                    #include <torch/torch.h>
                    #include <torch/csrc/jit/runtime/custom_operator.h>
                    int main() { return 0; }''',
                  lib = libnames,
                  libpath = libpath_torch,
                  cxxflags = map(lambda x: '-isystem' + x, (includes_torch_csrc_api + includes_torch)),
                  uselib_store="TORCH")
    # manually add the torch includes as system includes
    cfg.env['CXXFLAGS_TORCH'] += map(lambda x: '-isystem' + x, (includes_torch_csrc_api + includes_torch))


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    bld(
        target          = 'hxtorch_inc',
        export_includes = 'include',
    )

    bld(
        # pyext needed due to op registration, i.e. this can't be a shared lib
        # for non-Python usage
        features = 'cxx pyext',
        source = bld.path.ant_glob('src/hxtorch/**/*.cpp', excl='src/hxtorch/hxtorch.cpp'),
        target = 'hxtorch_cpp',
        use = ['hxtorch_inc', 'grenade_vx', 'TORCH_CPP'],
        install_path='${PREFIX}/lib',
        uselib = 'HXTORCH_LIBRARIES',
        rpath = bld.env.LIBPATH_TORCH,
    )

    bld(
        features = 'cxx cxxshlib pyext',
        source = 'src/hxtorch/hxtorch.cpp',
        target = '_hxtorch',
        use = ['hxtorch_cpp', 'grenade_vx', 'stadls_vx_v3', 'PYBIND11HXTORCH', 'TORCH'],
        defines = ['TORCH_EXTENSION_NAME=_hxtorch'],
        install_path='${PREFIX}/lib',
        uselib = 'HXTORCH_LIBRARIES',
        rpath = bld.env.LIBPATH_TORCH,
    )

    bld(
        target='hxtorch',
        features='py use',
        use=['pylogging', '_hxtorch', 'pygrenade_vx'],
        relative_trick=True,
        source=bld.path.ant_glob('src/pyhxtorch/**/*.py'),
        install_path = '${PREFIX}/lib',
        install_from='src/pyhxtorch',
    )

    bld(
        target='hxtorch_linting',
        features='py use pylint pycodestyle',
        use=['pylogging', '_hxtorch', 'pygrenade_vx'],
        relative_trick=True,
        source=bld.path.ant_glob('src/pyhxtorch/**/*.py'),
        pylint_config=os.path.join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=os.path.join(get_toplevel_path(), "code-format", "pycodestyle")
    )

    bld(
        target='hxtorch_hwtests',
        tests=bld.path.ant_glob('tests/hw/*.py'),
        features='use pytest',
        use=['hxtorch', 'dlens_vx_v3'],
        install_path='${PREFIX}/bin/tests/hw',
        test_timeout=1000,
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE
    )

    bld(
        target='hxtorch_swtests',
        tests=bld.path.ant_glob('tests/sw/*.py'),
        features='use pytest',
        use=['hxtorch'],
        install_path='${PREFIX}/bin/tests/sw',
    )

    bld(
        target = 'hxtorch_cpp_swtests',
        features = 'gtest cxx cxxprogram',
        source = bld.path.ant_glob('tests/sw/test-*.cpp'),
        use = ['hxtorch_cpp', 'GTEST'],
        linkflags = '-Wl,-z,defs',
        rpath = bld.env.LIBPATH_TORCH,
        install_path = '${PREFIX}/bin',
    )

    bld(
        name = 'mnist_model_state',
        features = 'install_task',
        install_to = '${PREFIX}/bin/tests/hw/',
        install_from = bld.path.ant_glob('mnist_model_state.pkl'),
        type = 'install_files',
        relative_trick=True,
    )

    bld(
        target = 'doxygen_hxtorch',
        features = 'doxygen',
        doxyfile = bld.root.make_node(os.path.join(get_toplevel_path(), "code-format", "doxyfile")),
        doxy_inputs = 'include/hxtorch',
        install_path = 'doc/hxtorch',
        pars = {
            "PROJECT_NAME": "\"hxtorch\"",
            "INCLUDE_PATH": os.path.join(get_toplevel_path(), "hxtorch", "include"),
            "OUTPUT_DIRECTORY": os.path.join(get_toplevel_path(), "build", "hxtorch", "doc")
        },
    )

    bld(
        target = 'doxygen_pyhxtorch',
        features = 'doxygen',
        doxyfile = bld.root.make_node(os.path.join(get_toplevel_path(), "code-format", "doxyfile")),
        doxy_inputs = 'src/pyhxtorch',
        install_path = 'doc/pyhxtorch',
        pars = {
            "PROJECT_NAME": "\"pyhxtorch\"",
            "OUTPUT_DIRECTORY": os.path.join(get_toplevel_path(), "build", "pyhxtorch", "doc")
        },
    )


# Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
