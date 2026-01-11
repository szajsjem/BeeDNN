
import glob
import re
import os

def get_construct_usage(content):
    match = re.search(r'std::string\s+\w+::constructUsage\(\)\s*\{.*?return\s*"(.*?)";', content, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def parse_usage(usage):
    lines = usage.split('\n')
    args_line = ""
    for line in reversed(lines):
        if ';' in line or ' ' not in line.strip():
            args_line = line.strip()
            break
    if not args_line:
        args_line = lines[-1].strip()
    
    parts = args_line.split(';')
    floats = []
    args_str = []
    
    for part in parts:
        part = part.strip()
        if not part: continue
        
        if part.startswith('i') or part.startswith('n'): # integer
            if 'In' in part or 'Rows' in part or 'Cols' in part: floats.append(10.0)
            elif 'Out' in part: floats.append(5.0)
            elif 'Kernel' in part: floats.append(3.0)
            elif 'Stride' in part: floats.append(1.0)
            elif 'Padding' in part: floats.append(1.0)
            elif 'Channel' in part: floats.append(3.0)
            else: floats.append(1.0)
        elif part.startswith('f'): # float
            if 'Rate' in part: floats.append(0.5)
            else: floats.append(0.1)
        elif part.startswith('s'): # string
            if 'Activation' in part: args_str.append('ReLU')
            elif 'Weight' in part: args_str.append('GlorotUniform')
            elif 'Bias' in part: args_str.append('Zeros')
            else: args_str.append('default')
        else:
            if 'size' in part.lower(): floats.append(10.0)
            else: floats.append(0.0)
            
    return floats, ';'.join(args_str)

files = glob.glob('src/Layer*.cpp')
calls = []

for f in files:
    with open(f, 'r') as file:
        content = file.read()
        basename = os.path.basename(f)
        classname = basename.replace('.cpp', '')
        
        # Skip Layer.cpp and LayerFactory.cpp
        if classname in ['Layer', 'LayerFactory']:
            continue
            
        usage = get_construct_usage(content)
        if usage:
            floats, str_arg = parse_usage(usage)
            floats_str = '{' + ', '.join([str(x) + 'f' for x in floats]) + '}'
            
            call = f'''
    {{
        std::cout << "Testing {classname}..." << std::endl;
        Layer* layer = LayerFactory::construct("{classname}", {floats_str}, "{str_arg}");
        if (layer) {{
            std::stringstream ss;
            layer->save(ss);
            if (ss.str().empty()) {{
                std::cout << "  FAILED: save produced empty output" << std::endl;
            }} else {{
                try {{
                    Layer* loaded = LayerFactory::loadLayer(ss);
                    if (loaded) {{
                        std::cout << "  PASSED" << std::endl;
                        delete loaded;
                    }} else {{
                        std::cout << "  FAILED: load returned null" << std::endl;
                    }}
                }} catch (const std::exception& e) {{
                    std::cout << "  FAILED: exception during load: " << e.what() << std::endl;
                }} catch (...) {{
                    std::cout << "  FAILED: unknown exception during load" << std::endl;
                }}
            }}
            delete layer;
        }} else {{
            std::cout << "  SKIPPED: could not construct (bad args?)" << std::endl;
        }}
    }}'''
            calls.append(call)

final_cpp = """
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include "allLayers.h"

using namespace beednn;

int main() {
    std::cout << "Starting Layer Save/Load Tests..." << std::endl;
""" + '\n'.join(calls) + """
    std::cout << "Finished Tests." << std::endl;
    return 0;
}
"""

with open('src/test_all_layers.cpp', 'w') as f:
    f.write(final_cpp)
    
print('Generated src/test_all_layers.cpp')
