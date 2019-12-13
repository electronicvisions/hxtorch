from waflib.extras.test_base import summary
import site
import os
import re

def depends(dep):
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
    libpath_torch = [os.path.join(x, 'torch/lib') for x in site_packages]
    libnames = []
    for fn in os.listdir(libpath_torch[0]):
        res = re.match('^lib(.+)\.so$', fn)
        libnames.append(res.group(1))

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
        features = 'cxx cxxshlib pyext pyembed',
        source = bld.path.ant_glob('src/hxtorch/*.cpp'),
        target = 'hxtorch',
        use = ['haldls_vx', 'lola_vx', 'grenade_vx', 'pyhxcomm_vx', 'pygrenade_vx', 'PYBIND11HXTORCH', 'TORCH'],
        linkflags = '-Wl,-z,defs',
        defines = ['TORCH_EXTENSION_NAME=hxtorch'],
        install_path='${PREFIX}/lib',
        uselib = 'HXTORCH_LIBRARIES',
        rpath = bld.env.LIBPATH_TORCH,
    )

    bld(
        target='hxtorch_hwtests',
        tests=bld.path.ant_glob('tests/hw/*.py'),
        features='use pytest',
        use=['hxtorch', 'dlens_vx'],
        install_path='${PREFIX}/bin',
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE
    )

    # Create test summary (to stdout and XML file)
    bld.add_post_fun(summary)
