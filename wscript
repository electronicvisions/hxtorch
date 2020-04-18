from waflib.extras.test_base import summary
import copy
import site
import os
import re

def depends(dep):
    dep('hxcomm')
    dep('haldls')
    dep('grenade')


def options(opt):
    opt.load('compiler_cxx')
    opt.load('test_base')
    opt.load('python')
    opt.load('pytest')


def configure(cfg):
    cfg.load('compiler_cxx')
    cfg.load('test_base')
    cfg.load('python')
    cfg.check_python_version()
    cfg.check_python_headers()
    cfg.load('pytest')

    cfg.check(
        compiler='cxx',
        features='cxx pyembed',
        uselib_store='PYBIND11HXTORCH',
        mandatory=True,
        header_name='pybind11/pybind11.h',
    )

    site_packages = site.getsitepackages()
    assert isinstance(site_packages, list) and len(site_packages) == 1
    includes_torch = [os.path.join(x, 'torch/include') for x in site_packages]
    includes_torch_csrc_api = [os.path.join(x, 'torch/include/torch/csrc/api/include') for x in site_packages]
    cfg.env.hxtorch_torch_includes = includes_torch + includes_torch_csrc_api
    libpath_torch = [os.path.join(x, 'torch/lib') for x in site_packages]
    libnames = []
    for fn in os.listdir(libpath_torch[0]):
        print(fn)
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
                  includes = includes_torch_csrc_api + includes_torch,
                  uselib_store="TORCH_CPP")

    cfg.check_cxx(fragment ='''
                    #include <torch/torch.h>
                    #include <torch/csrc/jit/runtime/custom_operator.h>
                    int main() { return 0; }''',
                  lib = libnames,
                  libpath = libpath_torch,
                  includes = includes_torch_csrc_api + includes_torch,
                  uselib_store="TORCH")


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    bld(
        target          = 'hxtorch_inc',
        export_includes = 'include',
    )

    bld(
        features = 'cxx cxxshlib pyembed',
        source = bld.path.ant_glob('src/hxtorch/**/*.cpp', excl='src/hxtorch/hxtorch.cpp'),
        target = 'hxtorch_cpp',
        use = ['hxtorch_inc', 'grenade_vx', 'TORCH_CPP'],
        install_path='${PREFIX}/lib',
        uselib = 'HXTORCH_LIBRARIES',
        rpath = bld.env.LIBPATH_TORCH,
    )

    bld(
        features = 'cxx cxxshlib pyext pyembed',
        source = 'src/hxtorch/hxtorch.cpp',
        target = 'hxtorch',
        use = ['hxtorch_cpp', 'hxtorch_pylibs', 'grenade_vx', 'stadls_vx', 'pyhxcomm_vx', 'pygrenade_vx', 'PYBIND11HXTORCH', 'TORCH'],
        linkflags = '-Wl,-z,defs',
        defines = ['TORCH_EXTENSION_NAME=hxtorch'],
        install_path='${PREFIX}/lib',
        uselib = 'HXTORCH_LIBRARIES',
        rpath = bld.env.LIBPATH_TORCH,
        cxxflags = ["-isystem" + e for e in bld.env.hxtorch_torch_includes],
    )

    bld(
        target='hxtorch_pylibs',
        features='py use',
        use='dlens_vx',
        relative_trick=True,
        source=bld.path.ant_glob('src/pyhxtorch/**/*.py'),
        install_path='${PREFIX}/lib/pyhxtorch',
        install_from='src/pyhxtorch',
    )

    bld(
        target='hxtorch_hwtests',
        tests=bld.path.ant_glob('tests/hw/*.py'),
        features='use pytest',
        use=['hxtorch', 'dlens_vx'],
        install_path='${PREFIX}/bin/tests/hw',
        test_timeout=300,
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
        features = 'gtest cxx cxxprogram pyembed',
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

# Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
