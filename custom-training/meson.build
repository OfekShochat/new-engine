project('sunset-training', 'cpp',
        default_options : ['cpp_std=c++17'])

cc = meson.get_compiler('cpp')
incdir = include_directories('..')
files = [
  'nn.cc',
  'trainer.cc',
  'optimizer.cc'
]

normalCompile = ['-std=c++17', '-fopenmp']

executable('train', 'main.cc', files,
           include_directories : incdir,
           cpp_args : normalCompile,
           link_args : '-fopenmp')