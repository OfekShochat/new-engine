project('chess-engine', 'cpp',
        default_options : ['cpp_std=c++17'])

cc = meson.get_compiler('cpp')

incdir = include_directories('../src')
files = []

files += [
    'src/search/search.cc',
    'src/neural/nn.cc',
    'src/uci.cc'
]

gdbCompile = ['-std=c++17', '-g']
normalCompile = '-std=c++17'

add_project_arguments(cc.get_supported_arguments(['-march=native']), language : 'cpp')

TTTest = executable('transpositionTest', 'src/tests/TTTest.cc', files,
                    include_directories : incdir, cpp_args : gdbCompile)
test('TT', TTTest)
NNTest = executable('neuralNetworkTest', 'src/tests/NNTest.cc', files,
                    include_directories : incdir, cpp_args : gdbCompile)
test('NN', NNTest)
NNTest = executable('LayerTest', 'src/tests/LayerTest.cc', files,
                    include_directories : incdir, cpp_args : gdbCompile)
test('Layer', NNTest)

executable('main', 'src/main.cc', files,
           include_directories : incdir,
           cpp_args : normalCompile)